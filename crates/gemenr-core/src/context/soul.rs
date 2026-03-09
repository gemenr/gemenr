use std::any::Any;
use std::fs;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::RwLock;
use tokio::task;
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
    /// Last observed file state for change detection.
    file_state: SoulFileState,
    /// Cached file modification time shared with lock-free readers.
    cached_mtime: Arc<AtomicU64>,
}

/// Shared state used to avoid taking the `SOUL.md` write lock when mtime is unchanged.
#[derive(Debug, Clone)]
pub(crate) struct SoulManagerState {
    manager: Arc<RwLock<SoulManager>>,
}

impl SoulManagerState {
    /// Create fast-path reload state from a shared manager handle.
    #[must_use]
    pub(crate) fn new(manager: Arc<RwLock<SoulManager>>) -> Self {
        Self { manager }
    }

    /// Return the latest `SOUL.md` content, reloading from disk only when mtime changes.
    pub(crate) async fn latest_content(&self) -> Result<String, SoulError> {
        let (path, cached_mtime, cached_content) = {
            let guard = self.manager.read().await;
            (
                guard.path.clone(),
                guard.cached_mtime.load(Ordering::Acquire),
                guard.content.clone(),
            )
        };

        let current_mtime = SoulFileState::read_mtime_async(path).await?;

        if current_mtime == cached_mtime {
            return Ok(cached_content);
        }

        let mut guard = self.manager.write().await;
        if current_mtime == guard.cached_mtime.load(Ordering::Acquire) {
            return Ok(guard.content().to_string());
        }

        guard.reload_from_disk_async().await?;
        Ok(guard.content().to_string())
    }
}

impl SoulManager {
    /// Load `SOUL.md` from the given workspace directory.
    ///
    /// If the file does not exist, a default template is created first.
    pub fn load(workspace: &Path) -> Result<Self, SoulError> {
        fs::create_dir_all(workspace)?;

        let path = workspace.join("SOUL.md");
        let snapshot = SoulDiskSnapshot::read_or_create_sync(path.clone())?;
        debug!(path = %path.display(), "loaded SOUL.md");

        Ok(Self {
            path,
            content: snapshot.content,
            file_state: snapshot.file_state.clone(),
            cached_mtime: Arc::new(AtomicU64::new(snapshot.file_state.modified_at_nanos)),
        })
    }

    /// Return the current `SOUL.md` content.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Reload `SOUL.md` from disk when the file changed.
    ///
    /// Returns `true` when the cached content was refreshed.
    pub fn reload_if_changed(&mut self) -> Result<bool, SoulError> {
        let current_mtime = SoulFileState::read_mtime_sync(self.path.clone())?;
        if current_mtime == self.cached_mtime.load(Ordering::Acquire) {
            return Ok(false);
        }

        self.load_latest()?;
        Ok(true)
    }

    /// Reload `SOUL.md` from disk when the file changed.
    ///
    /// Returns `true` when the cached content was refreshed.
    pub async fn reload_if_changed_async(&mut self) -> Result<bool, SoulError> {
        let current_mtime = SoulFileState::read_mtime_async(self.path.clone()).await?;
        if current_mtime == self.cached_mtime.load(Ordering::Acquire) {
            return Ok(false);
        }

        self.load_latest_async().await?;
        Ok(true)
    }

    /// Load the latest `SOUL.md` content from disk into memory.
    ///
    /// If the file was removed externally, the default template is recreated.
    pub fn load_latest(&mut self) -> Result<(), SoulError> {
        let snapshot = SoulDiskSnapshot::read_or_create_sync(self.path.clone())?;
        self.apply_snapshot(snapshot);
        debug!(path = %self.path.display(), "reloaded SOUL.md");
        Ok(())
    }

    /// Load the latest `SOUL.md` content from disk into memory.
    ///
    /// If the file was removed externally, the default template is recreated.
    pub async fn load_latest_async(&mut self) -> Result<(), SoulError> {
        let snapshot = SoulDiskSnapshot::read_or_create_async(self.path.clone()).await?;
        self.apply_snapshot(snapshot);
        debug!(path = %self.path.display(), "reloaded SOUL.md");
        Ok(())
    }

    /// Reload content from disk asynchronously.
    ///
    /// Returns `true` when the cached file state changed.
    pub async fn reload_from_disk_async(&mut self) -> Result<bool, SoulError> {
        let snapshot = SoulDiskSnapshot::read_or_create_async(self.path.clone()).await?;
        let changed = snapshot.content != self.content || snapshot.file_state != self.file_state;
        self.apply_snapshot(snapshot);
        debug!(path = %self.path.display(), "reloaded SOUL.md from disk");
        Ok(changed)
    }

    /// Replace the content of a section.
    ///
    /// The target section is identified by a markdown level-1 heading with the
    /// same name, for example `# Preferences`.
    pub fn update(&mut self, section: &str, content: &str) -> Result<(), SoulError> {
        self.apply_section_update(section, content)?;
        self.flush_sync()?;
        debug!(section, "updated SOUL.md section");
        Ok(())
    }

    /// Replace the content of a section and persist it through `spawn_blocking`.
    ///
    /// The target section is identified by a markdown level-1 heading with the
    /// same name, for example `# Preferences`.
    pub async fn update_async(&mut self, section: &str, content: &str) -> Result<(), SoulError> {
        self.apply_section_update(section, content)?;
        self.flush_async().await?;
        debug!(section, "updated SOUL.md section");
        Ok(())
    }

    /// Append an entry to a section.
    pub fn append(&mut self, section: &str, entry: &str) -> Result<(), SoulError> {
        self.apply_section_append(section, entry)?;
        self.flush_sync()?;
        debug!(section, "appended SOUL.md section entry");
        Ok(())
    }

    /// Append an entry to a section and persist it through `spawn_blocking`.
    pub async fn append_async(&mut self, section: &str, entry: &str) -> Result<(), SoulError> {
        self.apply_section_append(section, entry)?;
        self.flush_async().await?;
        debug!(section, "appended SOUL.md section entry");
        Ok(())
    }

    fn apply_snapshot(&mut self, snapshot: SoulDiskSnapshot) {
        self.content = snapshot.content;
        self.set_file_state(snapshot.file_state);
    }

    fn set_file_state(&mut self, file_state: SoulFileState) {
        self.cached_mtime
            .store(file_state.modified_at_nanos, Ordering::Release);
        self.file_state = file_state;
    }

    fn apply_section_update(&mut self, section: &str, content: &str) -> Result<(), SoulError> {
        let (body_start, body_end, has_next_section) =
            section_body_range(&self.content, section)
                .ok_or_else(|| SoulError::SectionNotFound(section.to_string()))?;
        let replacement = format_section_body(content, has_next_section);

        self.content
            .replace_range(body_start..body_end, &replacement);
        Ok(())
    }

    fn apply_section_append(&mut self, section: &str, entry: &str) -> Result<(), SoulError> {
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
        Ok(())
    }

    fn flush_sync(&mut self) -> Result<(), SoulError> {
        let file_state = SoulDiskSnapshot::write_sync(self.path.clone(), self.content.clone())?;
        self.set_file_state(file_state);
        Ok(())
    }

    async fn flush_async(&mut self) -> Result<(), SoulError> {
        let file_state =
            SoulDiskSnapshot::write_async(self.path.clone(), self.content.clone()).await?;
        self.set_file_state(file_state);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SoulFileState {
    modified_at_nanos: u64,
    size_bytes: u64,
}

impl SoulFileState {
    fn from_metadata(metadata: &fs::Metadata) -> Result<Self, SoulError> {
        Ok(Self {
            modified_at_nanos: system_time_to_unix_nanos(metadata.modified()?)?,
            size_bytes: metadata.len(),
        })
    }

    async fn read_mtime_async(path: PathBuf) -> Result<u64, SoulError> {
        task::spawn_blocking(move || match fs::metadata(&path) {
            Ok(metadata) => {
                SoulFileState::from_metadata(&metadata).map(|state| state.modified_at_nanos)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(0),
            Err(error) => Err(SoulError::Io(error)),
        })
        .await
        .map_err(join_error_to_soul_error)?
    }

    fn read_mtime_sync(path: PathBuf) -> Result<u64, SoulError> {
        run_future_sync(Self::read_mtime_async(path))
    }
}

#[derive(Debug, Clone)]
struct SoulDiskSnapshot {
    content: String,
    file_state: SoulFileState,
}

impl SoulDiskSnapshot {
    async fn read_or_create_async(path: PathBuf) -> Result<Self, SoulError> {
        task::spawn_blocking(move || {
            let content = match fs::read_to_string(&path) {
                Ok(content) => content,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    fs::write(&path, DEFAULT_SOUL_TEMPLATE)?;
                    DEFAULT_SOUL_TEMPLATE.to_string()
                }
                Err(error) => return Err(SoulError::Io(error)),
            };
            let metadata = fs::metadata(&path)?;
            let file_state = SoulFileState::from_metadata(&metadata)?;

            Ok(Self {
                content,
                file_state,
            })
        })
        .await
        .map_err(join_error_to_soul_error)?
    }

    fn read_or_create_sync(path: PathBuf) -> Result<Self, SoulError> {
        run_future_sync(Self::read_or_create_async(path))
    }

    async fn write_async(path: PathBuf, content: String) -> Result<SoulFileState, SoulError> {
        task::spawn_blocking(move || {
            fs::write(&path, &content)?;
            let metadata = fs::metadata(&path)?;
            SoulFileState::from_metadata(&metadata)
        })
        .await
        .map_err(join_error_to_soul_error)?
    }

    fn write_sync(path: PathBuf, content: String) -> Result<SoulFileState, SoulError> {
        run_future_sync(Self::write_async(path, content))
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

fn run_future_sync<T, Fut>(future: Fut) -> Result<T, SoulError>
where
    T: Send + 'static,
    Fut: Future<Output = Result<T, SoulError>> + Send + 'static,
{
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(SoulError::Io)?
            .block_on(future)
    })
    .join()
    .map_err(|panic_payload| {
        SoulError::Io(std::io::Error::other(format!(
            "SOUL async bridge panicked: {}",
            panic_payload_to_string(panic_payload.as_ref())
        )))
    })?
}

fn join_error_to_soul_error(error: task::JoinError) -> SoulError {
    SoulError::Io(std::io::Error::other(format!(
        "SOUL blocking task failed: {error}"
    )))
}

fn panic_payload_to_string(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn system_time_to_unix_nanos(system_time: SystemTime) -> Result<u64, SoulError> {
    let duration = system_time.duration_since(UNIX_EPOCH).map_err(|error| {
        SoulError::Io(std::io::Error::other(format!(
            "SOUL.md modified time is before UNIX_EPOCH: {error}"
        )))
    })?;
    u64::try_from(duration.as_nanos()).map_err(|error| {
        SoulError::Io(std::io::Error::other(format!(
            "SOUL.md modified time overflowed u64 nanoseconds: {error}"
        )))
    })
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;
    use std::time::{SystemTime, UNIX_EPOCH};

    use tokio::sync::RwLock;
    use tokio::time::timeout;

    use super::{DEFAULT_SOUL_TEMPLATE, SoulError, SoulManager, SoulManagerState};

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let directory = env::temp_dir().join(format!("gemenr-soul-{prefix}-{timestamp}"));

        fs::create_dir_all(&directory).expect("temp directory should be created");
        directory
    }

    fn rewrite_with_fresh_mtime(path: &Path, content: &str) {
        let original_mtime = fs::metadata(path)
            .expect("SOUL.md metadata should exist")
            .modified()
            .expect("SOUL.md mtime should exist");

        for _ in 0..30 {
            thread::sleep(Duration::from_millis(50));
            fs::write(path, content).expect("SOUL.md should be rewritten");

            let updated_mtime = fs::metadata(path)
                .expect("SOUL.md metadata should exist after rewrite")
                .modified()
                .expect("SOUL.md mtime should exist after rewrite");
            if updated_mtime > original_mtime {
                return;
            }
        }

        panic!("SOUL.md mtime did not advance after rewrite");
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

    #[test]
    fn reload_if_changed_observes_external_file_edit() {
        let directory = temp_dir("reload-external-edit");
        let soul_path = directory.join("SOUL.md");
        let mut manager = SoulManager::load(&directory).expect("SOUL.md should load");
        let updated_content = "# Identity\nupdated identity details\n\n# Preferences\nprefs\n\n# Experiences\nexp\n\n# Notes\nnotes\n";

        rewrite_with_fresh_mtime(&soul_path, updated_content);

        let reloaded = manager
            .reload_if_changed()
            .expect("reload should observe updated content");

        assert!(reloaded);
        assert_eq!(manager.content(), updated_content);
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[test]
    fn reload_if_changed_is_noop_when_file_unchanged() {
        let directory = temp_dir("reload-noop");
        let mut manager = SoulManager::load(&directory).expect("SOUL.md should load");
        let original = manager.content().to_string();

        let reloaded = manager
            .reload_if_changed()
            .expect("reload should succeed without refreshing");

        assert!(!reloaded);
        assert_eq!(manager.content(), original);
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn test_reload_no_change() {
        let directory = temp_dir("latest-content-no-change");
        let soul = Arc::new(RwLock::new(
            SoulManager::load(&directory).expect("SOUL.md should load"),
        ));
        let state = SoulManagerState::new(Arc::clone(&soul));
        let expected = soul.read().await.content().to_string();
        let read_guard = soul.read().await;

        let content = timeout(Duration::from_millis(100), state.latest_content())
            .await
            .expect("latest_content should not wait for a write lock")
            .expect("latest_content should succeed");

        assert_eq!(content, expected);
        drop(read_guard);
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn test_reload_after_external_modification() {
        let directory = temp_dir("latest-content-reload");
        let soul_path = directory.join("SOUL.md");
        let soul = Arc::new(RwLock::new(
            SoulManager::load(&directory).expect("SOUL.md should load"),
        ));
        let state = SoulManagerState::new(Arc::clone(&soul));
        let updated_content = "# Identity\nexternal update\n\n# Preferences\nprefs\n\n# Experiences\nexp\n\n# Notes\nnotes\n";

        rewrite_with_fresh_mtime(&soul_path, updated_content);

        let content = state
            .latest_content()
            .await
            .expect("latest_content should reload after external modification");

        assert_eq!(content, updated_content);
        assert_eq!(soul.read().await.content(), updated_content);
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn test_update_async_persists() {
        let directory = temp_dir("update-async");
        let soul_path = directory.join("SOUL.md");
        let mut manager = SoulManager::load(&directory).expect("SOUL.md should load");

        manager
            .update_async("Notes", "Remember accepted constraints.")
            .await
            .expect("update_async should succeed");

        let on_disk = fs::read_to_string(&soul_path).expect("SOUL.md should be readable");
        assert_eq!(on_disk, manager.content());
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let directory = temp_dir("latest-content-concurrent");
        let soul = Arc::new(RwLock::new(
            SoulManager::load(&directory).expect("SOUL.md should load"),
        ));
        let state = SoulManagerState::new(Arc::clone(&soul));
        let expected = soul.read().await.content().to_string();
        let mut handles = Vec::new();

        for _ in 0..8 {
            let state = state.clone();
            handles.push(tokio::spawn(async move { state.latest_content().await }));
        }

        for handle in handles {
            let content = handle
                .await
                .expect("task should complete")
                .expect("latest_content should succeed");
            assert_eq!(content, expected);
        }

        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }
}
