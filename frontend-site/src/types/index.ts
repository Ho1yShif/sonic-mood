/**
 * Type definitions for Sonic Mood
 * These types define the structure of data used throughout the application
 */

/**
 * Represents a single song recommendation
 */
export interface Song {
  /** Unique identifier for the song */
  id: string;
  /** The title of the song */
  title: string;
  /** The artist name */
  artist: string;
  /** Optional Spotify link - may be null if not available */
  spotifyLink?: string | null;
}

/**
 * API response structure from the backend recommendation endpoint
 * The backend uses pgvector to perform semantic search on song embeddings
 * and OpenAI to generate creative playlist names
 */
export interface RecommendationResponse {
  /** AI-generated playlist name that captures the vibe of the songs */
  playlistName: string;
  /** Array of exactly 5 song recommendations */
  recommendations: Song[];
}

/**
 * API error response structure
 */
export interface ApiError {
  /** Error message to display to the user */
  message: string;
  /** HTTP status code */
  status?: number;
  /** Error code for programmatic handling */
  code?: string;
}

/**
 * Loading states for the application
 */
export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

/**
 * Search request parameters
 */
export interface SearchRequest {
  /** Natural language query describing the desired music */
  query: string;
}

