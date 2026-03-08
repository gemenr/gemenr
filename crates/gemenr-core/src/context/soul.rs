use std::fs;
use std::path::{Path, PathBuf};

use tracing::debug;

const DEFAULT_SOUL_TEMPLATE: &str = "# Identity\n[Agent 的核心身份设定]\n\n# Preferences\n[用户偏好和工作习惯]\n\n# Experiences\n[从任务中积累的经验和教训]\n\n# Notes\n[其他需要记住的信息]\n";

/// Manages the `SOUL.md` file — the agent's persistent learning store.
///
/// `SOUL.md` is a markdown file with predefined sections whose content can be
/// injected into prompts and updated by runtime tools.
#[derive(Debug, Clone)]
pub struct SoulManager {
    /// Path to the `SOUL.md` file.
    path: PathBuf,
    /// Current content of `SOUL.md` cached in memory.
    content: String,
}

impl SoulManager {
    /// Load `SOUL.md` from the given workspace directory.
    ///
    /// If the file does not exist, a default template is created first.
    pub fn load(workspace: &Path) -> Result<Self, SoulError> {
        fs::create_dir_all(workspace)?;

        let path = workspace.join("SOUL.md");
        let content = if path.exists() {
            fs::read_to_string(&path)?
        } else {
            fs::write(&path, DEFAULT_SOUL_TEMPLATE)?;
            DEFAULT_SOUL_TEMPLATE.to_string()
        };

        debug!(path = %path.display(), "loaded SOUL.md");
        Ok(Self { path, content })
    }

    /// Return the current `SOUL.md` content.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Replace the content of a section.
    ///
    /// The target section is identified by a markdown level-1 heading with the
    /// same name, for example `# Preferences`.
    pub fn update(&mut self, section: &str, content: &str) -> Result<(), SoulError> {
        let (body_start, body_end, has_next_section) =
            section_body_range(&self.content, section)
                .ok_or_else(|| SoulError::SectionNotFound(section.to_string()))?;
        let replacement = format_section_body(content, has_next_section);

        self.content
            .replace_range(body_start..body_end, &replacement);
        self.flush()?;
        debug!(section, "updated SOUL.md section");
        Ok(())
    }

    /// Append an entry to a section.
    pub fn append(&mut self, section: &str, entry: &str) -> Result<(), SoulError> {
        let (body_start, body_end, has_next_section) =
            section_body_range(&self.content, section)
                .ok_or_else(|| SoulError::SectionNotFound(section.to_string()))?;

        let mut merged = self.content[body_start..body_end]
            .trim_matches('\n')
            .to_string();
        let trimmed_entry = entry.trim_matches('\n');

        if !merged.is_empty() && !trimmed_entry.is_empty() {
            merged.push('\n');
        }
        merged.push_str(trimmed_entry);

        let replacement = format_section_body(&merged, has_next_section);
        self.content
            .replace_range(body_start..body_end, &replacement);
        self.flush()?;
        debug!(section, "appended SOUL.md section entry");
        Ok(())
    }

    /// Flush the current content to disk.
    fn flush(&self) -> Result<(), SoulError> {
        fs::write(&self.path, &self.content)?;
        Ok(())
    }
}

/// Errors from `SOUL.md` operations.
#[derive(Debug, thiserror::Error)]
pub enum SoulError {
    /// I/O error while reading or writing `SOUL.md`.
    #[error("SOUL.md I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The requested section does not exist in `SOUL.md`.
    #[error("section not found: {0}")]
    SectionNotFound(String),
}

fn section_body_range(content: &str, section: &str) -> Option<(usize, usize, bool)> {
    let header = format!("# {section}");
    let header_start = content.find(&header)?;
    let body_start = match content[header_start..].find('\n') {
        Some(offset) => header_start + offset + 1,
        None => content.len(),
    };

    let remainder = &content[body_start..];
    let next_header_offset = remainder.find("\n# ");
    let body_end = next_header_offset
        .map(|offset| body_start + offset + 1)
        .unwrap_or(content.len());

    Some((body_start, body_end, next_header_offset.is_some()))
}

fn format_section_body(body: &str, has_next_section: bool) -> String {
    let trimmed = body.trim_matches('\n');

    if has_next_section {
        if trimmed.is_empty() {
            "\n".to_string()
        } else {
            format!("{trimmed}\n\n")
        }
    } else if trimmed.is_empty() {
        String::new()
    } else {
        format!("{trimmed}\n")
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{DEFAULT_SOUL_TEMPLATE, SoulError, SoulManager};

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let directory = env::temp_dir().join(format!("gemenr-soul-{prefix}-{timestamp}"));

        fs::create_dir_all(&directory).expect("temp directory should be created");
        directory
    }

    #[test]
    fn load_creates_default_template_when_file_is_missing() {
        let directory = temp_dir("load-default");
        let manager = SoulManager::load(&directory).expect("SOUL.md should load");
        let soul_path = directory.join("SOUL.md");

        assert_eq!(manager.content(), DEFAULT_SOUL_TEMPLATE);
        assert_eq!(
            fs::read_to_string(&soul_path).expect("SOUL.md should exist"),
            DEFAULT_SOUL_TEMPLATE
        );

        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[test]
    fn content_returns_full_file_contents() {
        let directory = temp_dir("content");
        let soul_path = directory.join("SOUL.md");
        let content =
            "# Identity\ncustom\n\n# Preferences\nprefs\n\n# Experiences\nexp\n\n# Notes\nnotes\n";
        fs::write(&soul_path, content).expect("SOUL.md should be written");

        let manager = SoulManager::load(&directory).expect("SOUL.md should load");

        assert_eq!(manager.content(), content);
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[test]
    fn update_replaces_section_content() {
        let directory = temp_dir("update");
        let mut manager = SoulManager::load(&directory).expect("SOUL.md should load");

        manager
            .update("Preferences", "Prefer concise progress updates.")
            .expect("section update should succeed");

        assert!(
            manager
                .content()
                .contains("# Preferences\nPrefer concise progress updates.\n\n# Experiences")
        );
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[test]
    fn append_adds_entry_to_section() {
        let directory = temp_dir("append");
        let mut manager = SoulManager::load(&directory).expect("SOUL.md should load");

        manager
            .append("Experiences", "- Prefer validating after each logical step")
            .expect("append should succeed");

        assert!(manager.content().contains("# Experiences\n[从任务中积累的经验和教训]\n- Prefer validating after each logical step\n\n# Notes"));
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[test]
    fn update_returns_error_for_missing_section() {
        let directory = temp_dir("missing-section");
        let mut manager = SoulManager::load(&directory).expect("SOUL.md should load");

        let error = manager
            .update("Missing", "value")
            .expect_err("missing section should error");

        assert!(matches!(error, SoulError::SectionNotFound(section) if section == "Missing"));
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[test]
    fn update_flushes_changes_to_disk() {
        let directory = temp_dir("flush");
        let soul_path = directory.join("SOUL.md");
        let mut manager = SoulManager::load(&directory).expect("SOUL.md should load");

        manager
            .update("Notes", "Remember accepted constraints.")
            .expect("update should succeed");

        let on_disk = fs::read_to_string(&soul_path).expect("SOUL.md should be readable");
        assert_eq!(on_disk, manager.content());

        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }
}
