"""
Live SOPBench environment loader.

Loads a domain's *raw* environment (no SOP constraint checking) so that an
agent's tool calls actually execute and mutate world state. The SOP itself is
NOT enforced by the env -- enforcement is the job of (a) the LLM, when the SOP
is in its prompt, or (b) ContrAgent, in the enforce condition. This mirrors
SOPBench's "prompt" env_mode and the Safety-Chip experimental design where the
constraints live outside the world simulator.

Faithful: tool schemas, descriptions and world state all come from the upstream
SOPBench domain/assistant modules; nothing is re-implemented here.
"""

import copy
import importlib
import inspect

# domain name -> (module suffix, domain class name)
DOMAIN_CLASSES = {
    "bank": "Bank",
    "dmv": "DMV",
    "healthcare": "Healthcare",
    "hotel": "Hotel",
    "library": "Library",
    "university": "University",
    "online_market": "OnlineMarket",
}


def load_modules(domain):
    dom = importlib.import_module(f"env.domains.{domain}.{domain}")
    asst = importlib.import_module(f"env.domains.{domain}.{domain}_assistant")
    return dom, asst


def build_env(domain, task):
    """Instantiate the raw (constraint-free) domain env from a task.

    Uses the task's initial_database as world state and constraint_parameters as
    the domain's tunable thresholds. dep_innate_full is the *none* dependency
    map keyed by the real class methods, so every action executes (the upstream
    default passes the string class name, which silently disables every method).
    """
    from env.helpers import get_domain_dependency_none

    dom, _ = load_modules(domain)
    cls = getattr(dom, DOMAIN_CLASSES[domain])
    data = copy.deepcopy(task["initial_database"])
    dep_params = copy.deepcopy(task.get("constraint_parameters", {}))

    sig = inspect.signature(cls.__init__)
    kwargs = {"data": data, "dep_innate_full": get_domain_dependency_none(cls)}
    if "dep_params" in sig.parameters and dep_params:
        kwargs["dep_params"] = dep_params
    return cls(**kwargs)


def tool_schemas(domain):
    """Gemini function_declarations for every assistant-exposed action."""
    _, asst = load_modules(domain)
    decls = []
    for a in asst.actions:
        decls.append(
            {
                "name": a["name"],
                # a few internal actions omit a description in the upstream
                # assistant; fall back to the name so the schema stays valid.
                "description": a.get("description") or a["name"],
                "parameters": _clean_schema(a["parameters"]),
            }
        )
    return decls


def _clean_schema(schema):
    """Gemini's schema dialect rejects a few JSON-Schema keywords SOPBench uses
    (additionalProperties, strict, anyOf with object branches). Strip/relax
    them without changing the parameter surface the model sees."""
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k in ("additionalProperties", "strict"):
            continue
        if k == "anyOf":
            # Gemini doesn't support anyOf. Every SOPBench anyOf is
            # "string (a password) OR {drivers_license_id, drivers_license_state}".
            # Collapsing to the string branch (the old behaviour) made it
            # impossible for the agent to authenticate a driver's-license user,
            # silently capping success. Instead MERGE both branches into one
            # optional-field object {password?, drivers_license_id?,
            # drivers_license_state?}; dispatch() normalises it back to the value
            # the env expects (a bare password string, or the DL dict).
            obj = next((b for b in v if b.get("type") == "object"), None)
            strb = next((b for b in v if b.get("type") == "string"), None)
            if obj is not None:
                props = {}
                if strb is not None:
                    props["password"] = {
                        "type": "string",
                        "description": strb.get("description", "account password"),
                    }
                props.update(obj.get("properties", {}))
                return _clean_schema({"type": "object", "properties": props})
            return _clean_schema(strb or v[0])
        if isinstance(v, dict):
            out[k] = _clean_schema(v)
        elif isinstance(v, list):
            out[k] = [_clean_schema(x) for x in v]
        else:
            out[k] = v
    return out


def dispatch(env, name, args):
    """Execute a tool call against the raw env. Returns (ok, payload).

    Domain methods return either a bool or a (bool, value) tuple. We normalise
    to (ok: bool, payload) where payload is the value or None.
    """
    if not hasattr(env, name):
        return False, f"unknown action: {name}"
    args = _normalize_identification(args)
    try:
        res = getattr(env, name)(**args)
    except Exception as e:  # noqa: BLE001 -- surface env errors to the agent
        return False, f"error: {type(e).__name__}: {e}"
    if isinstance(res, tuple):
        ok = bool(res[0])
        payload = res[1] if len(res) > 1 else None
        return ok, payload
    return bool(res), None


_ID_KEYS = ("identification", "identification_new")


def _normalize_identification(args):
    """Map the merged {password?, drivers_license_id?, drivers_license_state?}
    object the agent fills back to the value the env compares against: a bare
    password string, or the {drivers_license_id, drivers_license_state} dict."""
    out = dict(args)
    for k in _ID_KEYS:
        v = out.get(k)
        if isinstance(v, dict):
            if "drivers_license_id" in v:
                out[k] = {
                    "drivers_license_id": v.get("drivers_license_id"),
                    "drivers_license_state": v.get("drivers_license_state"),
                }
            elif "password" in v:
                out[k] = v["password"]
    return out


def observe_state(env):
    """Snapshot the live world state (the backend DB) for ContrAgent grounding."""
    return copy.deepcopy(getattr(env, "data", {}))
