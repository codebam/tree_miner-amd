use thiserror::Error;

/// Every journal failure. The C++ version throws a single `JournalError` carrying the
/// SQLite text; the variants here keep that information while letting callers distinguish
/// the two contract violations they can actually act on.
#[derive(Debug, Error)]
pub enum JournalError {
    #[error("journal: {context}: {source}")]
    Sqlite {
        context: String,
        #[source]
        source: rusqlite::Error,
    },

    /// Schema version the binary does not support, or a status/kind string in the database
    /// that no enumerator matches.
    #[error("journal: {0}")]
    Schema(String),

    /// Caller broke the interface contract (unknown id, or an attempt to persist
    /// `Submitting`).
    #[error("journal: {0}")]
    Contract(String),
}

impl JournalError {
    pub(crate) fn sqlite(context: impl Into<String>, source: rusqlite::Error) -> Self {
        JournalError::Sqlite {
            context: context.into(),
            source,
        }
    }
}

pub type Result<T> = std::result::Result<T, JournalError>;
