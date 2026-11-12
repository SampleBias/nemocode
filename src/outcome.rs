//! Typed execution evidence shared by the controller, journals, and evaluations.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Success,
    Failed,
    Unavailable,
    Cancelled,
    TimedOut,
    Partial,
    Stale,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChange {
    pub path: PathBuf,
    pub status: Status,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verification {
    pub key: String,
    pub tool: String,
    pub kind: String,
    pub status: Status,
    pub revisions: BTreeMap<PathBuf, String>,
    pub summary: String,
}

impl Verification {
    pub fn refresh(&mut self) {
        if self.status == Status::Success
            && (self.revisions.is_empty()
                || self.revisions.iter().any(|(path, expected)| {
                    !std::fs::read(path)
                        .is_ok_and(|bytes| crate::storage::hash(&bytes) == *expected)
                }))
        {
            self.status = Status::Stale;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutcome {
    pub schema_version: u32,
    pub status: Status,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    pub summary: String,
    pub data: Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub changes: Vec<FileChange>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification: Option<Verification>,
}

impl ToolOutcome {
    pub fn success(data: Value) -> Self {
        Self {
            schema_version: 1,
            status: Status::Success,
            error_code: None,
            summary: "Operation completed.".into(),
            data,
            changes: Vec::new(),
            verification: None,
        }
    }

    pub fn failure(status: Status, code: &str, summary: impl Into<String>) -> Self {
        Self {
            status,
            error_code: Some(code.into()),
            summary: summary.into(),
            ..Self::success(Value::Null)
        }
    }

    pub fn error(error: &anyhow::Error) -> Self {
        let message = format!("{error:#}");
        let (status, code) = if message.contains("stale edit") || message.contains("file changed") {
            (Status::Stale, "stale_revision")
        } else if message.contains("ambiguous edit") {
            (Status::Failed, "ambiguous_edit")
        } else if message.contains("snippet not found") {
            (Status::Failed, "snippet_not_found")
        } else if message.contains("not installed") || message.contains("not on PATH") {
            (Status::Unavailable, "missing_dependency")
        } else if message.contains("cancelled") {
            (Status::Cancelled, "cancelled")
        } else if message.contains("does not overwrite") || message.contains("already exists") {
            (Status::Failed, "already_exists")
        } else if message.contains("permission")
            || message.contains("escapes workspace")
            || message.contains("protected workspace")
            || message.contains("symlink escapes")
        {
            (Status::Blocked, "permission_denied")
        } else if error.chain().any(|e| {
            e.downcast_ref::<std::io::Error>()
                .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound)
        }) || message.contains("not a directory")
        {
            (Status::Failed, "not_found")
        } else {
            (Status::Failed, "tool_error")
        };
        Self::failure(status, code, message)
    }

    pub fn from_raw(name: &str, raw: &str) -> Self {
        let data: Value = serde_json::from_str(raw).unwrap_or_else(|_| json!(raw));
        if data["schema_version"] == 1
            && let Ok(outcome) = serde_json::from_value::<Self>(data.clone())
        {
            return outcome;
        }
        let mut result = Self::success(data);
        if result.data.get("exit_code").is_some() {
            let status = if result.data["cancelled"] == true {
                Status::Cancelled
            } else if result.data["timed_out"] == true {
                Status::TimedOut
            } else if result.data["exit_code"] == 0 {
                Status::Success
            } else if (matches!(name, "run_pytest" | "run_unittest")
                && result.data["exit_code"] == 5)
                || result.data["stderr"]
                    .as_str()
                    .is_some_and(|s| s.contains("No module named pytest"))
            {
                Status::Unavailable
            } else {
                Status::Failed
            };
            result.status = status;
            result.summary = format!("{name}: {status:?}; exit_code={}", result.data["exit_code"]);
            if status != Status::Success {
                result.error_code = Some(
                    match status {
                        Status::Cancelled => "cancelled",
                        Status::TimedOut => "timed_out",
                        Status::Unavailable if result.data["exit_code"] == 5 => {
                            "no_tests_collected"
                        }
                        Status::Unavailable => "missing_dependency",
                        _ => "check_failed",
                    }
                    .into(),
                );
            }
        } else if let Some(items) = result.data.as_array()
            && items.iter().any(|item| item.get("error").is_some())
        {
            result.status = Status::Partial;
            result.error_code = Some("partial_read".into());
            result.summary =
                "Some requested files could not be read; inspect per-file errors.".into();
        }
        result
    }

    pub fn json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn prompt_json(&self) -> anyhow::Result<String> {
        let mut value = serde_json::to_value(self)?;
        if let Some(verification) = value["verification"].as_object_mut() {
            let count = verification
                .remove("revisions")
                .and_then(|v| v.as_object().map(|m| m.len()))
                .unwrap_or(0);
            verification.insert("covered_files".into(), json!(count));
        }
        Ok(value.to_string())
    }

    pub fn is_success(&self) -> bool {
        self.status == Status::Success
    }
}

pub fn is_checker(name: &str) -> bool {
    matches!(
        name,
        "python_diagnostics" | "ruff_check" | "run_pytest" | "run_unittest"
    )
}

pub fn is_write(name: &str) -> bool {
    matches!(name, "create_file" | "create_multiple_files" | "edit_file")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subprocess_failures_are_not_successful_evidence() {
        for (data, status) in [
            (json!({"exit_code":1}), Status::Failed),
            (json!({"exit_code":0,"cancelled":true}), Status::Cancelled),
            (json!({"exit_code":0,"timed_out":true}), Status::TimedOut),
            (json!({"exit_code":5}), Status::Unavailable),
            (
                json!({"exit_code":1,"stderr":"No module named pytest"}),
                Status::Unavailable,
            ),
        ] {
            assert_eq!(
                ToolOutcome::from_raw("run_pytest", &data.to_string()).status,
                status
            );
        }
    }
}
