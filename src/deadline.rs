//! Request deadline propagation for Axon pipelines.
//!
//! Prevents wasted compute on requests that have already exceeded their
//! client-side timeout. The pipeline checks remaining budget at each
//! phase boundary and short-circuits with `DeadlineExceeded` if the
//! deadline has passed.
//!
//! Set via `X-Timeout-Ms` HTTP header (milliseconds).

use std::time::{Duration, Instant};
use std::fmt;

/// Budget tracking for a single request's deadline.
#[derive(Debug, Clone)]
pub struct RequestBudget {
    deadline: Instant,
}

impl RequestBudget {
    /// Create a budget from a timeout duration relative to now.
    pub fn from_timeout(timeout: Duration) -> Self {
        Self {
            deadline: Instant::now() + timeout,
        }
    }

    /// Returns the time remaining until deadline (saturates to zero if expired).
    pub fn remaining(&self) -> Duration {
        self.deadline
            .checked_duration_since(Instant::now())
            .unwrap_or(Duration::ZERO)
    }

    /// Returns true if the deadline has passed.
    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.deadline
    }

    /// Check if budget is still valid; returns error if deadline has passed.
    pub fn check(&self, phase: &str) -> Result<(), DeadlineExceeded> {
        if self.is_expired() {
            Err(DeadlineExceeded {
                phase: phase.to_string(),
            })
        } else {
            Ok(())
        }
    }
}

/// Deadline has been exceeded before entering a phase.
#[derive(Debug, Clone)]
pub struct DeadlineExceeded {
    /// Which phase was about to start when deadline expired (e.g. "pre", "model", "post").
    pub phase: String,
}

impl fmt::Display for DeadlineExceeded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "request deadline exceeded before {} phase", self.phase)
    }
}

impl std::error::Error for DeadlineExceeded {}

/// Parse deadline from HTTP header value (milliseconds).
///
/// Accepts a pure integer string representing milliseconds.
/// Returns None for non-numeric values.
pub fn parse_deadline_header(value: &str) -> Option<RequestBudget> {
    value
        .trim()
        .parse::<u64>()
        .ok()
        .map(|millis| RequestBudget::from_timeout(Duration::from_millis(millis)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_budget_not_expired() {
        let budget = RequestBudget::from_timeout(Duration::from_secs(10));
        assert!(!budget.is_expired());
    }

    #[test]
    fn test_budget_expired() {
        let budget = RequestBudget::from_timeout(Duration::from_millis(0));
        // Give it a tiny moment to ensure it's definitely expired
        thread::sleep(Duration::from_micros(10));
        assert!(budget.is_expired());
    }

    #[test]
    fn test_remaining_decreases() {
        let budget = RequestBudget::from_timeout(Duration::from_millis(100));
        let first = budget.remaining();
        thread::sleep(Duration::from_millis(10));
        let second = budget.remaining();
        assert!(first >= second);
    }

    #[test]
    fn test_check_ok() {
        let budget = RequestBudget::from_timeout(Duration::from_secs(10));
        assert!(budget.check("model").is_ok());
    }

    #[test]
    fn test_check_expired() {
        let budget = RequestBudget::from_timeout(Duration::from_millis(0));
        thread::sleep(Duration::from_micros(10));
        let result = budget.check("pre");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.phase, "pre");
        assert_eq!(err.to_string(), "request deadline exceeded before pre phase");
    }

    #[test]
    fn test_parse_header_millis() {
        let budget = parse_deadline_header("200").expect("should parse 200ms");
        assert!(!budget.is_expired());
    }

    #[test]
    fn test_parse_header_invalid() {
        let budget = parse_deadline_header("abc");
        assert!(budget.is_none());
    }
}
