// =============================================================================
// web_server/src/request.rs — HTTP request parser
// Ported from the General-Dynamics WS project, adapted for seal IP API
// Supports large bodies (base64-encoded images) up to 50 MB
// =============================================================================

use std::io::{BufRead, BufReader, Read};
use std::net::TcpStream;

/// Represents a parsed HTTP request
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<String>,
    pub body: Vec<u8>,
}

impl Request {
    /// Parse an HTTP request from a TCP stream.
    /// max_header_size: cap on total header bytes
    /// max_body_size: cap on Content-Length (50 MB default for base64 images)
    pub fn parse(
        stream: &mut TcpStream,
        max_header_size: usize,
        max_body_size: usize,
    ) -> Result<Request, RequestError> {
        let mut buf_reader = BufReader::new(stream);
        let mut headers = Vec::new();
        let mut header_size = 0;
        let mut cur_line = String::new();

        // Read headers line-by-line until blank line
        loop {
            cur_line.clear();
            let bytes_read = buf_reader
                .read_line(&mut cur_line)
                .map_err(|_| RequestError::IoError)?;

            if bytes_read == 0 || cur_line == "\r\n" {
                break;
            }

            header_size += bytes_read;
            if header_size > max_header_size {
                return Err(RequestError::HeaderTooLarge);
            }

            headers.push(cur_line.trim_end().to_string());
        }

        // Parse request line: "GET /path HTTP/1.1"
        let request_line = headers.first().ok_or(RequestError::BadRequest)?;
        let parts: Vec<&str> = request_line.split_whitespace().collect();
        let method = parts.get(0).ok_or(RequestError::BadRequest)?.to_string();
        let path = parts.get(1).ok_or(RequestError::BadRequest)?.to_string();

        if !matches!(
            method.as_str(),
            "GET" | "POST" | "PUT" | "DELETE" | "HEAD" | "OPTIONS" | "PATCH"
        ) {
            return Err(RequestError::MethodNotAllowed);
        }

        // Find Content-Length and read body
        let content_length = headers
            .iter()
            .find(|line| line.to_lowercase().starts_with("content-length:"))
            .and_then(|line| line.split(':').nth(1))
            .and_then(|val| val.trim().parse::<usize>().ok())
            .unwrap_or(0);

        if content_length > max_body_size {
            return Err(RequestError::BodyTooLarge);
        }

        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            buf_reader
                .read_exact(&mut body)
                .map_err(|_| RequestError::IoError)?;
        }

        Ok(Request {
            method,
            path,
            headers,
            body,
        })
    }

    /// Get body as a UTF-8 string (lossy)
    pub fn body_as_string(&self) -> String {
        String::from_utf8_lossy(&self.body).to_string()
    }

    /// Get a header value by name (case-insensitive)
    pub fn get_header(&self, name: &str) -> Option<String> {
        let name_lower = name.to_lowercase();
        for header in &self.headers {
            let header_lower = header.to_lowercase();
            if header_lower.starts_with(&format!("{}: ", name_lower)) {
                if let Some(value) = header.splitn(2, ": ").nth(1) {
                    return Some(value.to_string());
                }
            }
        }
        None
    }
}

/// Errors during request parsing
#[derive(Debug)]
pub enum RequestError {
    HeaderTooLarge,
    BodyTooLarge,
    MethodNotAllowed,
    BadRequest,
    IoError,
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            RequestError::HeaderTooLarge => write!(f, "Request headers too large"),
            RequestError::BodyTooLarge => write!(f, "Request body too large"),
            RequestError::MethodNotAllowed => write!(f, "HTTP method not allowed"),
            RequestError::BadRequest => write!(f, "Bad request"),
            RequestError::IoError => write!(f, "IO error"),
        }
    }
}

