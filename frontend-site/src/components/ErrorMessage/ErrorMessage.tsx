import { FiAlertCircle, FiRefreshCw } from 'react-icons/fi';
import type { ApiError } from '../../types';

/**
 * Props for the ErrorMessage component
 */
interface ErrorMessageProps {
  /** Error information to display */
  error: ApiError;
  /** Optional callback when user clicks retry button */
  onRetry?: () => void;
}

/**
 * ErrorMessage Component - Spotify Style
 * 
 * Clean error display with Spotify-inspired minimal design.
 */
export const ErrorMessage = ({ error, onRetry }: ErrorMessageProps) => {
  return (
    <div 
      className="
        bg-spotify-card
        rounded-lg p-8
        flex flex-col items-center justify-center gap-6
        text-center
        animate-fade-in
        max-w-lg mx-auto
      "
      role="alert"
      aria-live="assertive"
    >
      {/* Error icon */}
      <div 
        className="
          w-16 h-16
          bg-red-500/10
          rounded-full
          flex items-center justify-center
        "
        aria-hidden="true"
      >
        <FiAlertCircle className="text-3xl text-red-500" />
      </div>
      
      {/* Error message */}
      <div>
        <h3 className="text-xl font-bold text-white mb-2">
          Something went wrong
        </h3>
        <p className="text-spotify-text-subdued text-sm">
          {error.message}
        </p>
        
        {/* Show error code if available */}
        {error.code && (
          <p className="text-xs text-spotify-text-subdued mt-2 font-mono">
            Error: {error.code}
          </p>
        )}
      </div>
      
      {/* Retry button */}
      {onRetry && (
        <button
          onClick={onRetry}
          className="
            flex items-center gap-2
            px-6 py-3
            bg-white
            text-black
            rounded-full
            font-bold text-sm
            hover:scale-105
            focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-spotify-bg
            transition-all duration-200
          "
          aria-label="Retry the search"
        >
          <FiRefreshCw className="text-base" />
          <span>Try Again</span>
        </button>
      )}
    </div>
  );
};
