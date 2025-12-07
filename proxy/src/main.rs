use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};

// ============================================================================
// Data Models
// ============================================================================

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(flatten)]
    other: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct BrainAnalysisRequest {
    session_id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct BrainAnalysisResponse {
    surprise_score: f64,
    is_anomaly: bool,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    message: String,
}

// ============================================================================
// Application State
// ============================================================================

#[derive(Clone)]
struct AppState {
    groq_api_key: String,
    brain_url: String,
    http_client: reqwest::Client,
}

// ============================================================================
// Main Handler
// ============================================================================

async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(mut payload): Json<ChatCompletionRequest>,
) -> Response {
    // Extract the latest user message
    let user_message = match payload.messages.iter().rev().find(|m| m.role == "user") {
        Some(msg) => msg.content.clone(),
        None => {
            warn!("No user message found in request");
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "invalid_request".to_string(),
                    message: "No user message found in messages array".to_string(),
                }),
            )
                .into_response();
        }
    };

    // Generate session ID (using a simple hash of the first message for demo)
    let session_id = format!(
        "session_{}",
        payload
            .messages
            .first()
            .map(|m| format!("{:x}", md5::compute(&m.content)))
            .unwrap_or_else(|| "default".to_string())
    );

    info!(
        "Processing request - Session: {}, Message length: {}",
        &session_id[..16],
        user_message.len()
    );

    // ========================================================================
    // SECURITY AUDIT: Query the Python Brain
    // ========================================================================

    let brain_request = BrainAnalysisRequest {
        session_id: session_id.clone(),
        text: user_message.clone(),
    };

    let brain_response = match state
        .http_client
        .post(&format!("{}/analyze", state.brain_url))
        .json(&brain_request)
        .send()
        .await
    {
        Ok(resp) => match resp.json::<BrainAnalysisResponse>().await {
            Ok(analysis) => analysis,
            Err(e) => {
                error!("Failed to parse brain response: {}", e);
                // Fail-safe: block on error
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ErrorResponse {
                        error: "brain_error".to_string(),
                        message: "Security analysis service unavailable".to_string(),
                    }),
                )
                    .into_response();
            }
        },
        Err(e) => {
            error!("Failed to contact brain service: {}", e);
            // Fail-safe: block on error
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: "brain_unavailable".to_string(),
                    message: "Security analysis service unavailable".to_string(),
                }),
            )
                .into_response();
        }
    };

    info!(
        "Brain analysis - Session: {}, Surprise: {:.4}, Anomaly: {}",
        &session_id[..16],
        brain_response.surprise_score,
        brain_response.is_anomaly
    );

    // ========================================================================
    // DECISION: Block if anomalous
    // ========================================================================

    if brain_response.is_anomaly {
        warn!(
            "BLOCKED - Anomalous request detected! Session: {}, Surprise: {:.4}",
            &session_id[..16],
            brain_response.surprise_score
        );
        return (
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "security_violation".to_string(),
                message: format!(
                    "Request blocked due to anomalous pattern (surprise score: {:.2})",
                    brain_response.surprise_score
                ),
            }),
        )
            .into_response();
    }

    // ========================================================================
    // GROQ FORWARDING: Safe path
    // ========================================================================

    // Ensure we're using a Groq-compatible model
    if payload.model.is_none() {
        payload.model = Some("llama-3.1-8b-instant".to_string());
        info!("No model specified, defaulting to llama-3.1-8b-instant");
    }

    info!(
        "Forwarding to Groq - Model: {}",
        payload.model.as_ref().unwrap()
    );

    let groq_response = match state
        .http_client
        .post("https://api.groq.com/openai/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", state.groq_api_key))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status();
            let body = match resp.text().await {
                Ok(text) => text,
                Err(e) => {
                    error!("Failed to read Groq response body: {}", e);
                    return (
                        StatusCode::BAD_GATEWAY,
                        Json(ErrorResponse {
                            error: "groq_error".to_string(),
                            message: "Failed to read response from LLM provider".to_string(),
                        }),
                    )
                        .into_response();
                }
            };

            if !status.is_success() {
                error!("Groq API error - Status: {}, Body: {}", status, body);
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(ErrorResponse {
                        error: "groq_api_error".to_string(),
                        message: format!("LLM provider returned error: {}", status),
                    }),
                )
                    .into_response();
            }

            body
        }
        Err(e) => {
            error!("Failed to contact Groq API: {}", e);
            return (
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse {
                    error: "groq_unavailable".to_string(),
                    message: "Failed to contact LLM provider".to_string(),
                }),
            )
                .into_response();
        }
    };

    info!("Successfully forwarded request to Groq and received response");

    // Return Groq's response as-is
    (
        StatusCode::OK,
        [("Content-Type", "application/json")],
        groq_response,
    )
        .into_response()
}

// ============================================================================
// Health Check
// ============================================================================

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "mnemosyne-proxy"
    }))
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() {
    println!("DEBUG: Starting main function...");
    
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    println!("DEBUG: Tracing initialized. Loading environment...");
    info!("Starting Mnemosyne Proxy...");

    // Load environment variables
    dotenv::dotenv().ok();

    println!("DEBUG: Checking GROQ_API_KEY...");
    let groq_api_key = std::env::var("GROQ_API_KEY").expect("GROQ_API_KEY must be set");
    println!("DEBUG: GROQ_API_KEY found (length: {})", groq_api_key.len());
    let brain_url = std::env::var("BRAIN_URL").unwrap_or_else(|_| "http://mnemosyne-brain:5000".to_string());

    info!("Configuration loaded - Brain URL: {}", brain_url);

    // Create shared state
    let state = Arc::new(AppState {
        groq_api_key,
        brain_url,
        http_client: reqwest::Client::new(),
    });

    // Build router
    let app = Router::new()
        .route("/chat/completions", post(chat_completions_handler))
        .route("/health", axum::routing::get(health_check))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Start server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
        .await
        .expect("Failed to bind to port 8080");

    info!("Mnemosyne Proxy listening on 0.0.0.0:8080");

    axum::serve(listener, app)
        .await
        .expect("Server failed to start");
}
