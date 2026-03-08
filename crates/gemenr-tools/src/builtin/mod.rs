//! Built-in tool handlers available in Phase 1.

/// Built-in file read tool.
pub mod fs_read;
/// Built-in file write tool.
pub mod fs_write;
/// Built-in shell tool.
pub mod shell;
/// Built-in SOUL update tool.
pub mod update_soul;

use std::sync::Arc;

use gemenr_core::SoulManager;
use tokio::sync::RwLock;

use crate::ToolPlane;

/// Register all built-in Phase 1 tools into the provided tool plane.
pub fn register_builtin_tools(plane: &mut ToolPlane, soul: Arc<RwLock<SoulManager>>) {
    plane.register(
        shell::shell_spec(),
        Box::new(shell::ShellHandler::default()),
    );
    plane.register(fs_read::fs_read_spec(), Box::new(fs_read::FsReadHandler));
    plane.register(
        fs_write::fs_write_spec(),
        Box::new(fs_write::FsWriteHandler),
    );
    plane.register(
        update_soul::update_soul_spec(),
        Box::new(update_soul::UpdateSoulHandler::new(soul)),
    );
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use gemenr_core::SoulManager;
    use tokio::sync::RwLock;

    use super::register_builtin_tools;
    use crate::ToolPlane;

    fn temp_dir(prefix: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("gemenr-builtin-{prefix}-{timestamp}"));

        fs::create_dir_all(&directory).expect("temp directory should be created");
        directory
    }

    #[test]
    fn registers_all_builtin_tools() {
        let directory = temp_dir("register");
        let soul = Arc::new(RwLock::new(
            SoulManager::load(&directory).expect("SOUL.md should load"),
        ));
        let mut plane = ToolPlane::new();

        register_builtin_tools(&mut plane, soul);

        let names = plane
            .list()
            .tools
            .into_iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["fs.read", "fs.write", "shell", "update_soul"]);
        fs::remove_dir_all(directory).expect("temp directory should be removed");
    }
}
