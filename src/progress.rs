//! Bounded detection of repeated outcomes, independent of model wording.
use crate::{ToolCall, outcome::ToolOutcome};
use serde_json::Value;
use std::{collections::HashMap, path::Path};

#[derive(Default)]
pub struct Progress {
    seen: HashMap<String, (String, u32)>,
    pub blocked: u32,
}

pub fn normalized_arguments(arguments: &str, cwd: &Path) -> Value {
    fn normalize(value: &mut Value, key: &str, cwd: &Path) {
        match value {
            Value::String(text)
                if matches!(
                    key,
                    "file_path" | "path" | "working_directory" | "script" | "paths" | "file_paths"
                ) =>
            {
                if let Ok(path) = crate::normalize_path_with_base(text, cwd) {
                    *text = path.display().to_string();
                }
            }
            Value::Array(items) => {
                for item in items {
                    normalize(item, key, cwd);
                }
            }
            Value::Object(fields) => {
                for (key, item) in fields {
                    normalize(item, key, cwd);
                }
            }
            _ => {}
        }
    }
    let mut value = serde_json::from_str(arguments).unwrap_or(Value::Null);
    normalize(&mut value, "", cwd);
    value
}

impl Progress {
    pub fn clear(&mut self) {
        self.seen.clear();
        self.blocked = 0;
    }

    pub fn record(&mut self, call: &ToolCall, cwd: &Path, outcome: &ToolOutcome) -> Option<String> {
        if outcome
            .changes
            .iter()
            .any(|change| change.status == crate::outcome::Status::Success)
        {
            // A real mutation makes a fresh attempt meaningful, even when the
            // subsequent test has the same exit code as the previous attempt.
            self.clear();
        }
        // An actual file revision or current directory change is new evidence.
        let key = format!(
            "{}:{}:{}",
            cwd.display(),
            call.function.name,
            normalized_arguments(&call.function.arguments, cwd)
        );
        let mut evidence = if outcome.is_success() {
            outcome.data.clone()
        } else {
            Value::Null
        };
        if let Some(fields) = evidence.as_object_mut() {
            fields.remove("duration_ms");
        }
        let signature = crate::storage::hash(
            serde_json::json!({
                "status":outcome.status,"error_code":outcome.error_code,
                "data":evidence,"changes":outcome.changes,
                "checked_revisions":outcome.verification.as_ref().map(|check|&check.revisions),
            })
            .to_string()
            .as_bytes(),
        );
        if self.seen.len() >= 128 && !self.seen.contains_key(&key) {
            self.seen.clear();
        }
        let entry = self.seen.entry(key).or_insert((signature.clone(), 0));
        if entry.0 != signature {
            *entry = (signature, 0);
        }
        entry.1 += 1;
        if entry.1 >= 3 {
            self.blocked += 1;
            Some(format!(
                "This operation has returned the same outcome {} times. Choose a different concrete action using the recorded result. Current directory: {}. Do not repeat this call without new evidence.",
                entry.1,
                cwd.display()
            ))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::{FileChange, Status, Verification};
    #[test]
    fn changed_revisions_and_successful_edits_allow_new_attempts() {
        let mut progress = Progress::default();
        let call = ToolCall {
            function: crate::FunctionCall {
                name: "run_unittest".into(),
                arguments: "{}".into(),
            },
            ..Default::default()
        };
        let cwd = Path::new("/tmp");
        let mut failure = ToolOutcome::failure(Status::Failed, "check_failed", "failed");
        for revision in ["first", "second", "third", "fourth"] {
            failure.verification = Some(Verification {
                key: "check".into(),
                tool: "run_unittest".into(),
                kind: "behavior".into(),
                status: Status::Failed,
                revisions: [("code.py".into(), revision.into())].into(),
                summary: "failed".into(),
            });
            assert!(progress.record(&call, cwd, &failure).is_none());
        }
        assert!(progress.record(&call, cwd, &failure).is_none());
        assert!(progress.record(&call, cwd, &failure).is_some());
        let mut edit = ToolOutcome::success(Value::Null);
        edit.changes.push(FileChange {
            path: "code.py".into(),
            status: Status::Success,
            sha256: Some("fifth".into()),
            error: None,
        });
        progress.record(&call, cwd, &edit);
        assert_eq!(progress.blocked, 0);
        assert!(progress.record(&call, cwd, &failure).is_none());
    }
}
