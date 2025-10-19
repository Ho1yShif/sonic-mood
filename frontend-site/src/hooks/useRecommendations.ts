import { useState, useCallback } from 'react';
import type { RecommendationResponse, ApiError, LoadingState } from '../types';
import { getRecommendations } from '../services/api';

/**
 * Custom hook for managing song recommendation API calls and state
 * 
 * This hook encapsulates the logic for:
 * - Making API calls to fetch recommendations
 * - Managing loading, success, and error states
 * - Handling errors gracefully
 * - Providing a clean interface for components
 * 
 * Usage example:
 * ```tsx
 * const { data, error, loading, fetchRecommendations } = useRecommendations();
 * 
 * const handleSearch = async (query: string) => {
 *   await fetchRecommendations(query);
 * };
 * ```
 * 
 * @returns Object containing data, error state, loading state, and fetch function
 */
export const useRecommendations = () => {
  // State for storing the API response data
  const [data, setData] = useState<RecommendationResponse | null>(null);
  
  // State for storing error information
  const [error, setError] = useState<ApiError | null>(null);
  
  // State for tracking the current loading state
  const [loadingState, setLoadingState] = useState<LoadingState>('idle');
  
  /**
   * Fetches song recommendations based on a natural language query
   * 
   * This function:
   * 1. Clears previous errors and sets loading state
   * 2. Calls the API service to get recommendations
   * 3. Updates state based on success or failure
   * 4. Returns success/failure status
   * 
   * @param query - Natural language description of desired songs
   * @returns Promise<boolean> - true if successful, false if error occurred
   */
  const fetchRecommendations = useCallback(async (query: string): Promise<boolean> => {
    // Reset error state and set loading
    setError(null);
    setLoadingState('loading');
    
    try {
      // Call the API service
      const response = await getRecommendations(query);
      
      // Update state with successful response
      setData(response);
      setLoadingState('success');
      return true;
      
    } catch (err) {
      // Handle errors from the API
      const apiError = err as ApiError;
      setError(apiError);
      setData(null);
      setLoadingState('error');
      return false;
    }
  }, []);
  
  /**
   * Resets the hook state to initial values
   * Useful for clearing results after a new search or on component unmount
   */
  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoadingState('idle');
  }, []);
  
  return {
    /** The recommendation data (null if no data or error) */
    data,
    /** Error information (null if no error) */
    error,
    /** Current loading state */
    loadingState,
    /** Whether the API call is currently in progress */
    isLoading: loadingState === 'loading',
    /** Whether the last call was successful */
    isSuccess: loadingState === 'success',
    /** Whether the last call resulted in an error */
    isError: loadingState === 'error',
    /** Whether the hook is in its initial state */
    isIdle: loadingState === 'idle',
    /** Function to fetch recommendations */
    fetchRecommendations,
    /** Function to reset the hook state */
    reset,
  };
};

