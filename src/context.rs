//! Structured, bounded tool evidence. Exact source lines are never middle-cut.
use serde_json::{Value, json};

fn shrink(value: &mut Value, budget: usize) {
    match value {
        Value::Object(fields) => {
            if fields.get("start_line").and_then(Value::as_u64).is_some()
                && let Some(content) = fields.get("content").and_then(Value::as_str)
            {
                let mut used = 0;
                let lines: Vec<_> = content
                    .lines()
                    .take_while(|line| {
                        used += line.len() + 1;
                        used <= budget
                    })
                    .collect();
                if lines.len() < content.lines().count() {
                    let next = fields["start_line"].as_u64().unwrap() + lines.len() as u64;
                    let content = lines.join("\n");
                    fields.insert("content".into(), json!(content));
                    fields.insert("next_line".into(), json!(next));
                    fields.insert("more".into(), json!(true));
                    fields.insert("truncated".into(), json!(true));
                }
            }
            for (key, item) in fields.iter_mut() {
                if matches!(key.as_str(), "path" | "sha256" | "file_path" | "content") {
                    continue;
                }
                shrink(item, budget);
            }
        }
        Value::Array(items) => {
            let share = budget / items.len().max(1);
            for item in items.iter_mut() {
                shrink(item, share.max(100));
            }
            if items.len() > 1 && serde_json::to_string(items).is_ok_and(|s| s.len() > budget) {
                items.truncate(items.len().div_ceil(2));
            }
        }
        Value::String(text) if text.len() > budget => {
            *text = crate::truncate_middle(
                text,
                budget / 4,
                "evidence; use read_tool_output for full output",
            );
        }
        _ => {}
    }
}

pub fn bound(prompt_json: &str, limit: usize, result_id: &str) -> String {
    let mut value: Value = serde_json::from_str(prompt_json).expect("serialized outcome");
    if prompt_json.len() <= limit {
        return prompt_json.into();
    }
    value["truncated"] = json!(true);
    value["result_id"] = json!(result_id);
    for divisor in [2, 4, 8, 16, 32] {
        shrink(&mut value["data"], (limit / divisor).max(64));
        if let Some(summary) = value["summary"].as_str() {
            value["summary"] = json!(crate::truncate_middle(summary, 400, "summary"));
        }
        let encoded = value.to_string();
        if encoded.len() <= limit {
            return encoded;
        }
    }
    // Metadata remains truthful even when a single long path/record cannot fit.
    json!({"schema_version":1,"status":value["status"],"error_code":value["error_code"],
        "summary":"Output exceeds the context allowance. Retrieve the relevant field with read_tool_output.",
        "truncated":true,"result_id":result_id,"data":null}).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unicode_batch_outputs_have_an_aggregate_byte_limit() {
        let data:Vec<_> = (0..8).map(|i|json!({"path":format!("file-{i}.py"),"sha256":"a".repeat(64),
            "start_line":1,"next_line":201,"content":(1..201).map(|n|format!("{n}: # {}\n","🦀".repeat(40))).collect::<String>()})).collect();
        let prompt = json!({"schema_version":1,"status":"partial","error_code":"partial_read","summary":"one file unavailable","data":data}).to_string();
        for budget in [1024, 2048, 4096] {
            let result = bound(&prompt, budget, "id");
            assert!(result.len() <= budget);
            let value: Value = serde_json::from_str(&result).unwrap();
            assert_eq!(value["status"], "partial");
            assert_eq!(value["error_code"], "partial_read");
            assert_eq!(value["truncated"], true);
        }
    }
    #[test]
    fn preserves_read_lines_and_cursor_with_a_hard_size_bound() {
        let lines = (1..200)
            .map(|i| format!("{i}: value_{i} = 'some complete source text'\n"))
            .collect::<String>();
        let original = json!({"schema_version":1,"status":"success","summary":"read",
            "data":{"path":"code.py","sha256":"revision","start_line":1,"next_line":200,"content":lines,"more":false}});
        let text = bound(&original.to_string(), 1600, "test-result");
        assert!(text.len() <= 1600);
        let value: Value = serde_json::from_str(&text).unwrap();
        let content = value["data"]["content"].as_str().unwrap();
        assert!(
            content
                .lines()
                .all(|s| s.ends_with("'some complete source text'"))
        );
        assert_eq!(value["data"]["next_line"], content.lines().count() + 1);
        assert_eq!(value["result_id"], "test-result");
        assert_eq!(value["data"]["more"], true);
    }
}
