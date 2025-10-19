import type { RecommendationResponse, ApiError } from '../types';
import { allMockResponses } from '../mocks/data';

/**
 * Mock API service for development and testing
 * This simulates the backend API behavior with realistic delays and error scenarios
 * 
 * The mock service helps developers:
 * - Test the frontend without a running backend
 * - Simulate various API responses and error states
 * - Develop and iterate quickly on the UI
 */

/**
 * Simulates an API delay to mimic real network latency
 * @param ms - Milliseconds to delay
 */
const delay = (ms: number): Promise<void> => {
  return new Promise(resolve => setTimeout(resolve, ms));
};

/**
 * Determines which mock response to return based on the query content
 * This provides a more realistic testing experience by varying responses
 * 
 * @param query - The user's natural language query
 * @returns The most appropriate mock response for the query
 */
const selectMockResponse = (query: string): RecommendationResponse => {
  const lowerQuery = query.toLowerCase();
  
  // Match keywords to appropriate mock responses
  if (lowerQuery.includes('workout') || lowerQuery.includes('gym') || lowerQuery.includes('exercise') || lowerQuery.includes('energetic')) {
    return allMockResponses[0]; // Workout response
  }
  
  if (lowerQuery.includes('relax') || lowerQuery.includes('chill') || lowerQuery.includes('study') || lowerQuery.includes('focus') || lowerQuery.includes('calm')) {
    return allMockResponses[1]; // Relaxing response
  }
  
  if (lowerQuery.includes('sad') || lowerQuery.includes('heartbreak') || lowerQuery.includes('cry') || lowerQuery.includes('emotional')) {
    return allMockResponses[2]; // Sad response
  }
  
  if (lowerQuery.includes('happy') || lowerQuery.includes('upbeat') || lowerQuery.includes('cheerful') || lowerQuery.includes('joyful')) {
    return allMockResponses[3]; // Happy response
  }
  
  if (lowerQuery.includes('party') || lowerQuery.includes('dance') || lowerQuery.includes('club')) {
    return allMockResponses[4]; // Party response
  }
  
  // Default response for generic queries
  return allMockResponses[5];
};

/**
 * Fetches mock song recommendations based on a natural language query
 * 
 * Simulates the backend's behavior:
 * - Network latency (500-1000ms)
 * - Semantic search using pgvector (mocked)
 * - AI-generated playlist names (mocked)
 * - Occasional errors for testing error handling
 * 
 * @param query - Natural language description of desired songs
 * @returns Promise containing playlist name and song recommendations
 * @throws ApiError if the request fails (simulated)
 */
export const getMockRecommendations = async (query: string): Promise<RecommendationResponse> => {
  // Simulate network delay (500-1000ms)
  const delayMs = Math.random() * 500 + 500;
  await delay(delayMs);
  
  // Simulate occasional network errors (5% chance) for testing error handling
  if (Math.random() < 0.05) {
    const error: ApiError = {
      message: 'Network error: Failed to fetch recommendations. Please try again.',
      status: 500,
      code: 'NETWORK_ERROR'
    };
    throw error;
  }
  
  // Validate query
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
  
  // Return appropriate mock response based on query content
  return selectMockResponse(query);
};

