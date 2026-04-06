// =============================================================================
// web_server/src/response.rs — HTTP response builder (Builder Pattern)
// Ported from the General-Dynamics WS project, extended with binary body
// support for returning JPEG images, and CORS headers for browser access.
// =============================================================================

/// Represents a fully-built HTTP response ready to send over TCP
pub struct Response {
    status_code: u16,
    status_text: String,
    content_type: String,
    extra_headers: Vec<String>,
    body: Vec<u8>,
}

impl Response {
    // --- Status code constructors ---
    pub fn ok() -> ResponseBuilder {
        ResponseBuilder::new(200, "OK")
    }
    pub fn created() -> ResponseBuilder {
        ResponseBuilder::new(201, "Created")
    }
    pub fn bad_request() -> ResponseBuilder {
        ResponseBuilder::new(400, "Bad Request")
    }
    pub fn not_found() -> ResponseBuilder {
        ResponseBuilder::new(404, "Not Found")
    }
    pub fn method_not_allowed() -> ResponseBuilder {
        ResponseBuilder::new(405, "Method Not Allowed")
    }
    pub fn payload_too_large() -> ResponseBuilder {
        ResponseBuilder::new(413, "Payload Too Large")
    }
    pub fn header_too_large() -> ResponseBuilder {
        ResponseBuilder::new(431, "Request Header Fields Too Large")
    }
    pub fn internal_server_error() -> ResponseBuilder {
        ResponseBuilder::new(500, "Internal Server Error")
    }
    pub fn service_unavailable() -> ResponseBuilder {
        ResponseBuilder::new(503, "Service Unavailable")
    }

    /// Serialize response into raw bytes for writing to the TCP stream
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut header_str = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\n",
            self.status_code,
            self.status_text,
            self.content_type,
            self.body.len()
        );
        for h in &self.extra_headers {
            header_str.push_str(h);
            header_str.push_str("\r\n");
        }
        header_str.push_str("\r\n");

        let mut bytes = header_str.into_bytes();
        bytes.extend_from_slice(&self.body);
        bytes
    }

    /// Legacy string-based serialization for simple text responses
    pub fn to_http_string(&self) -> String {
        // Only safe for text bodies
        let body_str = String::from_utf8_lossy(&self.body);
        format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
            self.status_code,
            self.status_text,
            self.content_type,
            self.body.len(),
            body_str
        )
    }

    pub fn status_code(&self) -> u16 {
        self.status_code
    }
}

/// Builder for constructing responses fluently
pub struct ResponseBuilder {
    status_code: u16,
    status_text: String,
    content_type: String,
    extra_headers: Vec<String>,
    body: Vec<u8>,
}

impl ResponseBuilder {
    fn new(status_code: u16, status_text: &str) -> Self {
        ResponseBuilder {
            status_code,
            status_text: status_text.to_string(),
            content_type: "text/html".to_string(),
            extra_headers: Vec::new(),
            body: Vec::new(),
        }
    }

    /// Set HTML body
    pub fn html(mut self, html: String) -> Self {
        self.body = html.into_bytes();
        self.content_type = "text/html; charset=utf-8".to_string();
        self
    }

    /// Set plain text body
    pub fn text(mut self, text: String) -> Self {
        self.body = text.into_bytes();
        self.content_type = "text/plain; charset=utf-8".to_string();
        self
    }

    /// Set JSON body
    pub fn json(mut self, json: String) -> Self {
        self.body = json.into_bytes();
        self.content_type = "application/json".to_string();
        self
    }

    /// Set raw binary body with a specific content type (e.g. image/jpeg)
    pub fn binary(mut self, data: Vec<u8>, content_type: &str) -> Self {
        self.body = data;
        self.content_type = content_type.to_string();
        self
    }

    /// Add an extra header
    pub fn header(mut self, header: String) -> Self {
        self.extra_headers.push(header);
        self
    }

    /// Build the final Response
    pub fn build(self) -> Response {
        Response {
            status_code: self.status_code,
            status_text: self.status_text,
            content_type: self.content_type,
            extra_headers: self.extra_headers,
            body: self.body,
        }
    }
}

