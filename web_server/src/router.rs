// =============================================================================
// web_server/src/router.rs — Strategy Pattern Router
// Ported from the General-Dynamics WS project.
// Maps (method, path_pattern) → Handler function.
// =============================================================================

use crate::request::Request;
use crate::response::Response;
use std::panic::RefUnwindSafe;

/// A Handler is a function that takes a Request and returns a Response.
/// Each handler is a different "strategy" for handling a specific route.
pub type Handler = Box<dyn Fn(&Request) -> Response + Send + Sync + RefUnwindSafe>;

struct Route {
    method: String,
    path: String,
    handler: Handler,
}

/// Router stores routes and dispatches requests to the correct handler
pub struct Router {
    routes: Vec<Route>,
}

impl Router {
    pub fn new() -> Self {
        Router { routes: Vec::new() }
    }

    pub fn get(&mut self, path: &str, handler: Handler) {
        self.add_route("GET", path, handler);
    }

    pub fn post(&mut self, path: &str, handler: Handler) {
        self.add_route("POST", path, handler);
    }

    pub fn options(&mut self, path: &str, handler: Handler) {
        self.add_route("OPTIONS", path, handler);
    }

    fn add_route(&mut self, method: &str, path: &str, handler: Handler) {
        self.routes.push(Route {
            method: method.to_string(),
            path: path.to_string(),
            handler,
        });
    }

    /// Dispatch a request to the matching handler, or return 404
    pub fn handle(&self, request: &Request) -> Response {
        // Handle CORS preflight for any path
        if request.method == "OPTIONS" {
            return Response::ok()
                .text(String::new())
                .build();
        }

        // Strip query string for matching
        let match_path = request.path.split('?').next().unwrap_or(&request.path);

        for route in &self.routes {
            if route.method == request.method && self.path_matches(&route.path, match_path) {
                return (route.handler)(request);
            }
        }

        // No route found
        Response::not_found()
            .json(format!(
                r#"{{"error":"Not found","path":"{}","method":"{}"}}"#,
                request.path, request.method
            ))
            .build()
    }

    /// Check if a request path matches a route pattern.
    /// Supports path parameters like /api/users/:id
    fn path_matches(&self, route_pattern: &str, request_path: &str) -> bool {
        let route_parts: Vec<&str> = route_pattern.split('/').collect();
        let request_parts: Vec<&str> = request_path.split('/').collect();

        if route_parts.len() != request_parts.len() {
            return false;
        }

        for (rp, rq) in route_parts.iter().zip(request_parts.iter()) {
            if rp.starts_with(':') {
                continue; // wildcard segment
            }
            if rp != rq {
                return false;
            }
        }
        true
    }
}

impl Default for Router {
    fn default() -> Self {
        Router::new()
    }
}

