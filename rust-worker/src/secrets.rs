//! Mirrors `buffdata/server/worker.py::secrets_for_project()` exactly: the provider-secrets
//! file is either the original flat shape (one shared credential set) or a per-project
//! nested shape (`{"alpha": {...}, "beta": {...}}`), so pooled projects on different
//! provider accounts don't silently share -- or leak -- each other's keys.

use std::collections::BTreeMap;

use serde_json::Value;

pub const ALLOWED_SECRETS: &[&str] = &[
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
];

pub fn secrets_for_project(
    all_secrets: &Value,
    project: &str,
) -> Result<BTreeMap<String, String>, String> {
    let object = match all_secrets {
        Value::Null => return Ok(BTreeMap::new()),
        Value::Object(map) if map.is_empty() => return Ok(BTreeMap::new()),
        Value::Object(map) => map,
        _ => return Err("Invalid provider-secret file".to_string()),
    };

    let all_dicts = object.values().all(|v| v.is_object());
    let all_strings = object.values().all(|v| v.is_string());

    let selected: BTreeMap<String, Value> = if all_dicts {
        object
            .get(project)
            .and_then(|v| v.as_object())
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default()
    } else if all_strings {
        object.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    } else {
        return Err("Invalid provider-secret file".to_string());
    };

    let mut values = BTreeMap::new();
    for (key, value) in selected {
        if !ALLOWED_SECRETS.contains(&key.as_str()) {
            return Err("Invalid provider-secret file".to_string());
        }
        match value {
            Value::String(s) => {
                values.insert(key, s);
            }
            _ => return Err("Invalid provider-secret file".to_string()),
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_or_missing_file_yields_no_secrets() {
        assert!(secrets_for_project(&Value::Null, "alpha")
            .unwrap()
            .is_empty());
        assert!(secrets_for_project(&json!({}), "alpha").unwrap().is_empty());
    }

    #[test]
    fn flat_shape_is_shared_across_every_project() {
        let all = json!({"GEMINI_API_KEY": "shared-key"});
        let alpha = secrets_for_project(&all, "alpha").unwrap();
        let beta = secrets_for_project(&all, "beta").unwrap();
        assert_eq!(alpha.get("GEMINI_API_KEY").unwrap(), "shared-key");
        assert_eq!(beta.get("GEMINI_API_KEY").unwrap(), "shared-key");
    }

    #[test]
    fn nested_shape_isolates_each_project_and_never_leaks_the_other() {
        let all = json!({
            "alpha": {"GEMINI_API_KEY": "alpha-key"},
            "beta": {"OPENAI_API_KEY": "beta-key"}
        });
        let alpha = secrets_for_project(&all, "alpha").unwrap();
        assert_eq!(alpha.get("GEMINI_API_KEY").unwrap(), "alpha-key");
        assert!(!alpha.contains_key("OPENAI_API_KEY"));

        let beta = secrets_for_project(&all, "beta").unwrap();
        assert_eq!(beta.get("OPENAI_API_KEY").unwrap(), "beta-key");
        assert!(!beta.contains_key("GEMINI_API_KEY"));
    }

    #[test]
    fn nested_shape_project_with_no_entry_gets_nothing() {
        let all = json!({"alpha": {"GEMINI_API_KEY": "alpha-key"}});
        assert!(secrets_for_project(&all, "gamma").unwrap().is_empty());
    }

    #[test]
    fn disallowed_key_is_rejected_even_if_the_value_is_a_string() {
        let all = json!({"NOT_A_REAL_PROVIDER_KEY": "x"});
        assert!(secrets_for_project(&all, "alpha").is_err());
    }

    #[test]
    fn mixed_dict_and_string_values_at_the_top_level_is_rejected() {
        let all = json!({"alpha": {"GEMINI_API_KEY": "x"}, "OPENAI_API_KEY": "y"});
        assert!(secrets_for_project(&all, "alpha").is_err());
    }

    #[test]
    fn non_string_leaf_value_is_rejected() {
        let all = json!({"GEMINI_API_KEY": 12345});
        assert!(secrets_for_project(&all, "alpha").is_err());
    }
}
