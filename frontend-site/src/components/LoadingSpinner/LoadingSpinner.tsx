/**
 * Props for the LoadingSpinner component
 */
interface LoadingSpinnerProps {
  /** Optional message to display below the spinner */
  message?: string;
}

/**
 * LoadingSpinner Component - Spotify Style
 * 
 * Simple, clean loading indicator matching Spotify's minimal aesthetic.
 */
export const LoadingSpinner = ({ message = "Finding your perfect songs..." }: LoadingSpinnerProps) => {
  return (
    <div 
      className="flex flex-col items-center justify-center py-20"
      role="status"
      aria-live="polite"
    >
      {/* Simple spinner */}
      <div className="relative w-16 h-16 mb-6">
        {/* Spinning ring */}
        <div 
          className="
            absolute inset-0
            border-4 border-spotify-elevated border-t-primary-purple
            rounded-full
            animate-spin
          "
          aria-hidden="true"
        />
      </div>
      
      {/* Loading message */}
      {message && (
        <div className="text-center">
          <p className="text-lg font-semibold mb-1 animate-color-fade">
            {message}
          </p>
          <p className="text-sm animate-color-fade">
            This won't take long
          </p>
        </div>
      )}
      
      {/* Screen reader text */}
      <span className="sr-only">Loading recommendations</span>
    </div>
  );
};
