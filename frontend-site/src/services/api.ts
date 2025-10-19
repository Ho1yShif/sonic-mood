import type { RecommendationResponse, ApiError, Song } from '../types';
import { getMockRecommendations } from './mockApi';

/**
 * API service for communicating with the backend
 * 
 * The backend uses:
 * - Render Postgres with pgvector for semantic search on song embeddings
 * - OpenAI API for generating creative playlist names
 * 
 * This service can switch between mock data (for development) and real API calls
 * based on the VITE_USE_MOCK_DATA environment variable.
 */

// Backend response structure (snake_case)
interface BackendSong {
  title: string;
  artist: string;
  spotifyLink: string | null;
}

interface BackendResponse {
  playlist_name: string;
  songs: BackendSong[];
}

// Get API configuration from environment variables
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000/api';
// Default to mock data for development (set VITE_USE_MOCK_DATA=false to use real API)
const USE_MOCK_DATA = import.meta.env.VITE_USE_MOCK_DATA !== 'false';

/**
 * Fetches song recommendations from the backend API using natural language input
 * 
 * The backend workflow:
 * 1. Receives the natural language query
 * 2. Generates an embedding vector for the query using OpenAI
 * 3. Uses pgvector to perform semantic similarity search on song embeddings in Postgres
 * 4. Returns the top 5 most similar songs
 * 5. Generates a creative playlist name using OpenAI based on the songs
 * 
 * @param query - Natural language description of desired songs (max 500 chars)
 * @returns Promise containing playlist name and song recommendations
 * @throws ApiError if the request fails
 */
export const getRecommendations = async (query: string): Promise<RecommendationResponse> => {
  // Use mock data if enabled
  if (USE_MOCK_DATA) {
    return getMockRecommendations(query);
  }
  
  // Validate query before making API call
  if (!query || query.trim().length === 0) {
    const error: ApiError = {
      message: 'Please enter a description of the music you\'re looking for.',
      status: 400,
      code: 'INVALID_QUERY'
    };
    throw error;
  }
  
  if (query.length > 500) {
    const error: ApiError = {
      message: 'Query is too long. Please keep it under 500 characters.',
      status: 400,
      code: 'QUERY_TOO_LONG'
    };
    throw error;
  }
  
  try {
    // Make API request to the backend
    const response = await fetch(`${API_URL}/recommendations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query }),
    });
    
    // Handle non-OK responses
    if (!response.ok) {
      let errorMessage = 'Failed to fetch recommendations. Please try again.';
      
      // Try to parse error message from response
      try {
        const errorData = await response.json();
        errorMessage = errorData.message || errorMessage;
      } catch {
        // If parsing fails, use default message
      }
      
      const error: ApiError = {
        message: errorMessage,
        status: response.status,
        code: response.status === 404 ? 'NOT_FOUND' : 
              response.status === 500 ? 'SERVER_ERROR' : 
              'UNKNOWN_ERROR'
      };
      throw error;
    }
    
    // Parse backend response
    const backendData: BackendResponse = await response.json();
    
    // Transform backend response to frontend format
    const transformedData: RecommendationResponse = {
      playlistName: backendData.playlist_name,
      recommendations: backendData.songs.map((song, index): Song => ({
        id: `${song.artist}-${song.title}-${index}`.replace(/\s+/g, '-').toLowerCase(),
        title: song.title,
        artist: song.artist,
        spotifyLink: song.spotifyLink || null,
      })),
    };
    
    return transformedData;
    
  } catch (error) {
    // Handle network errors or other exceptions
    if ((error as ApiError).status) {
      // Re-throw API errors
      throw error;
    }
    
    // Handle generic network errors
    const apiError: ApiError = {
      message: 'Network error: Unable to reach the server. Please check your connection.',
      status: 0,
      code: 'NETWORK_ERROR'
    };
    throw apiError;
  }
};

