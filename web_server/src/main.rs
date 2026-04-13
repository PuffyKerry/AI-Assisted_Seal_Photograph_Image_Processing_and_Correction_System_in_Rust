// =============================================================================
// web_server/src/main.rs — Entry point for the Seal IP Web Server
//
// Architecture ported from the General-Dynamics Packet Sniffer & Web Server
// project (same thread pool, strategy-pattern router, TCP handling).
//
// Exposes the AI-Assisted Seal Photograph Image Processing functions as a
// REST-ish JSON API with an interactive browser UI.
//
// Usage:
//   cargo run -p web_server                   # Starts on 0.0.0.0:8080
//   cargo run -p web_server -- --port 3000    # Custom port
//
// Then open http://localhost:8080 in a browser to use the upload UI,
// or POST JSON to /api/dehaze, /api/clahe, /api/gamma, /api/process.
// =============================================================================

mod request;
mod response;
mod router;
mod handlers;
mod convert;

use std::{
    collections::VecDeque,
    net::{TcpListener, TcpStream},
    panic,
    sync::{Arc, Condvar, Mutex},
    thread,
    time::Duration,
};

use request::{Request, RequestError};
use response::Response;
use router::Router;

// =============================================================================
// Thread Pool (ported from GD WS — same architecture)
// =============================================================================
type Job = Box<dyn FnOnce() + Send + panic::UnwindSafe + 'static>;

enum Message {
    Job(Job),
    Terminate,
}

struct ThreadPool {
    workers: Vec<Worker>,
    shared: Arc<Shared>,
}

struct Shared {
    state: Mutex<SharedState>,
    available: Condvar,
}

struct SharedState {
    queue: VecDeque<Message>,
    queue_size: usize,
    reserved: usize,
    shutting_down: bool,
}

struct Reservation {
    shared: Arc<Shared>,
    used: bool,
}

impl Reservation {
    fn submit<F>(mut self, f: F)
    where
        F: FnOnce() + Send + 'static + panic::UnwindSafe,
    {
        let mut guard = match self.shared.state.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.reserved = guard.reserved.saturating_sub(1);
        guard.queue.push_back(Message::Job(Box::new(f)));
        self.shared.available.notify_one();
        self.used = true;
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.used {
            return;
        }
        if let Ok(mut guard) = self.shared.state.lock() {
            guard.reserved = guard.reserved.saturating_sub(1);
        }
    }
}

#[derive(Debug)]
enum TryReserveError {
    QueueFull,
    ShuttingDown,
}

impl ThreadPool {
    fn new(size: usize, queue_size: usize) -> ThreadPool {
        assert!(size > 0);
        assert!(queue_size > 0);
        let shared = Arc::new(Shared {
            state: Mutex::new(SharedState {
                queue: VecDeque::with_capacity(queue_size.min(1024)),
                queue_size,
                reserved: 0,
                shutting_down: false,
            }),
            available: Condvar::new(),
        });

        let mut workers = Vec::with_capacity(size);
        for id in 0..size {
            workers.push(Worker::new(id, Arc::clone(&shared)));
        }
        ThreadPool { workers, shared }
    }

    fn try_reserve(&self) -> Result<Reservation, TryReserveError> {
        let mut guard = self
            .shared
            .state
            .lock()
            .map_err(|_| TryReserveError::ShuttingDown)?;
        if guard.shutting_down {
            return Err(TryReserveError::ShuttingDown);
        }
        if guard.queue.len() + guard.reserved >= guard.queue_size {
            return Err(TryReserveError::QueueFull);
        }
        guard.reserved += 1;
        Ok(Reservation {
            shared: Arc::clone(&self.shared),
            used: false,
        })
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        let mut guard = match self.shared.state.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.shutting_down = true;
        for _ in 0..self.workers.len() {
            guard.queue.push_back(Message::Terminate);
        }
        self.shared.available.notify_all();
        drop(guard);
        for w in &mut self.workers {
            if let Some(handle) = w.thread.take() {
                let _ = handle.join();
            }
        }
    }
}

struct Worker {
    _id: usize,
    thread: Option<thread::JoinHandle<()>>,
}

impl Worker {
    fn new(id: usize, shared: Arc<Shared>) -> Worker {
        let thread = thread::spawn(move || loop {
            let message = {
                let mut guard = match shared.state.lock() {
                    Ok(g) => g,
                    Err(poisoned) => poisoned.into_inner(),
                };
                while guard.queue.is_empty() && !guard.shutting_down {
                    guard = match shared.available.wait(guard) {
                        Ok(g) => g,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                }
                if guard.shutting_down && guard.queue.is_empty() {
                    return;
                }
                guard.queue.pop_front()
            };

            match message {
                Some(Message::Job(job)) => {
                    if let Err(e) = panic::catch_unwind(job) {
                        eprintln!("[Worker {}] Handler panicked: {:?}", id, e);
                    }
                }
                Some(Message::Terminate) | None => return,
            }
        });
        Worker {
            _id: id,
            thread: Some(thread),
        }
    }
}

// =============================================================================
// Main — parse args, build router, start server
// =============================================================================
fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Parse optional --port flag
    let mut port: u16 = 8080;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" | "-p" => {
                if i + 1 < args.len() {
                    port = args[i + 1].parse().unwrap_or(8080);
                    i += 2;
                } else {
                    eprintln!("--port requires a value");
                    i += 1;
                }
            }
            "--help" | "-h" => {
                println!("Seal Photo Processing Web Server");
                println!("Usage: cargo run -p web_server -- [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --port, -p PORT   Port to listen on (default: 8080)");
                println!("  --help, -h        Show this help");
                println!();
                println!("API Endpoints:");
                println!("  GET  /             Interactive upload UI");
                println!("  GET  /api/health   Health check");
                println!("  POST /api/dehaze   DCP dehazing");
                println!("  POST /api/clahe    CLAHE contrast enhancement");
                println!("  POST /api/gamma    Gamma brightness correction");
                println!("  POST /api/process  Full pipeline (DCP+CLAHE+Gamma)");
                return Ok(());
            }
            _ => {
                i += 1;
            }
        }
    }

    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr)?;

    // Thread pool sizing
    let cores = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    // Image processing is CPU-heavy, so use 2× cores (not 4× like pure I/O servers)
    let thread_count = (cores * 2).clamp(4, 32);
    let queue_size = thread_count * 8;
    let pool = ThreadPool::new(thread_count, queue_size);

    // === Build Router (Strategy Pattern) ===
    let mut router = Router::new();

    // Interactive UI
    router.get("/", Box::new(handlers::home_handler));

    // API endpoints
    router.get("/api/health", Box::new(handlers::health_handler));
    router.post("/api/dehaze", Box::new(handlers::dehaze_handler));
    router.post("/api/clahe", Box::new(handlers::clahe_handler));
    router.post("/api/gamma", Box::new(handlers::gamma_handler));
    router.post("/api/process", Box::new(handlers::process_handler));

    let router = Arc::new(router);

    println!("========================================");
    println!("  🦭 Seal Photo Processing Web Server");
    println!("  AI-Assisted Image Processing System");
    println!("========================================");
    println!();
    println!("  Listening on: http://localhost:{}", port);
    println!("  Thread pool:  {} workers, queue {}", thread_count, queue_size);
    println!("  CPU cores:    {}", cores);
    println!();

    // Eagerly load the CNN model so the user sees status at startup
    print!("  CNN model:    ");
    handlers::init_cnn_model();
    println!();

    println!("  Endpoints:");
    println!("    GET  /             → Interactive upload UI");
    println!("    GET  /api/health   → Health check");
    println!("    POST /api/dehaze   → DCP dehazing");
    println!("    POST /api/clahe    → CLAHE contrast enhancement");
    println!("    POST /api/gamma    → Gamma brightness correction");
    println!("    POST /api/process  → Full pipeline (DCP+CLAHE+Gamma)");
    println!();
    println!("  Open http://localhost:{} in your browser!", port);
    println!("========================================");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[Server] Error accepting connection: {}", e);
                continue;
            }
        };

        let router = Arc::clone(&router);

        match pool.try_reserve() {
            Ok(reservation) => {
                reservation.submit(move || {
                    handle_connection(stream, &router);
                });
            }
            Err(TryReserveError::QueueFull) => {
                let resp = Response::service_unavailable()
                    .json(r#"{"error":"Server busy, please retry"}"#.to_string())
                    .build();
                use std::io::Write;
                let _ = stream.write_all(&resp.to_bytes());
            }
            Err(TryReserveError::ShuttingDown) => {
                let resp = Response::service_unavailable()
                    .json(r#"{"error":"Server shutting down"}"#.to_string())
                    .build();
                use std::io::Write;
                let _ = stream.write_all(&resp.to_bytes());
            }
        }
    }

    Ok(())
}

// =============================================================================
// Connection handler — parse request, route, send response
// =============================================================================
fn handle_connection(mut stream: TcpStream, router: &Router) {
    // Large limits for base64-encoded images: 8 KB headers, 50 MB body
    const MAX_HEADER_SIZE: usize = 8 * 1024;
    const MAX_BODY_SIZE: usize = 50 * 1024 * 1024;

    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(60)));

    let request = match Request::parse(&mut stream, MAX_HEADER_SIZE, MAX_BODY_SIZE) {
        Ok(req) => req,
        Err(e) => {
            let response = match e {
                RequestError::HeaderTooLarge => Response::header_too_large()
                    .json(r#"{"error":"Headers too large"}"#.to_string())
                    .build(),
                RequestError::BodyTooLarge => Response::payload_too_large()
                    .json(r#"{"error":"Payload too large (max 50MB)"}"#.to_string())
                    .build(),
                RequestError::MethodNotAllowed => Response::method_not_allowed()
                    .json(r#"{"error":"Method not allowed"}"#.to_string())
                    .build(),
                RequestError::BadRequest | RequestError::IoError => Response::bad_request()
                    .json(r#"{"error":"Bad request"}"#.to_string())
                    .build(),
            };
            use std::io::Write;
            let _ = stream.write_all(&response.to_bytes());
            return;
        }
    };

    // Log the request
    println!(
        "[{}] {} {} (body: {} bytes)",
        chrono_now(),
        request.method,
        request.path,
        request.body.len()
    );

    let response = router.handle(&request);

    use std::io::Write;
    let _ = stream.write_all(&response.to_bytes());
}

/// Simple timestamp without pulling in chrono crate
fn chrono_now() -> String {
    use std::time::SystemTime;
    match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => {
            let secs = d.as_secs();
            let hours = (secs / 3600) % 24;
            let mins = (secs / 60) % 60;
            let s = secs % 60;
            format!("{:02}:{:02}:{:02}", hours, mins, s)
        }
        Err(_) => "??:??:??".to_string(),
    }
}

