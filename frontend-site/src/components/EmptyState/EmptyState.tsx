import { FiSearch } from 'react-icons/fi';

/**
 * Props for the EmptyState component
 */
interface EmptyStateProps {
  /** Callback function called when a user clicks on an example query */
  onSearch: (query: string) => void;
}

/**
 * EmptyState Component - Spotify Style
 * 
 * Clean, minimal welcome screen with Spotify-inspired design.
 * Focuses on simplicity and clarity.
 * 
 * @param props - EmptyState props
 */
export const EmptyState = ({ onSearch }: EmptyStateProps) => {
  const exampleQueries = [
    'upbeat songs for a morning workout',
    'relaxing jazz for studying',
    'sad indie songs about heartbreak',
    'energetic dance music for a party',
    'calm acoustic songs for meditation',
  ];
  
  return (
    <div 
      className="flex flex-col items-center justify-center py-16 px-4 text-center animate-fade-in"
      role="region"
      aria-label="Welcome message"
    >
      {/* Main message */}
      <h2 className="text-4xl sm:text-5xl font-black text-white mb-4">
        Discover your new favorite playlist
      </h2>
      
      <p className="text-spotify-text-subdued text-base sm:text-lg max-w-2xl mb-12">
        Tell us your vibe, and we'll find the perfect soundtrack
      </p>
      
      {/* Example queries */}
      <div className="max-w-2xl w-full">
        <div className="bg-spotify-card rounded-lg p-6">
          <h3 className="text-base font-bold text-white mb-4 flex items-center justify-center gap-2">
            <FiSearch className="text-spotify-text-subdued" />
            Try searching for:
          </h3>
          
          <div className="space-y-2">
            {exampleQueries.map((query, index) => (
              <button
                key={index}
                onClick={() => onSearch(query)}
                className="
                  w-full
                  bg-spotify-elevated
                  hover:bg-spotify-card-hover
                  px-4 py-3
                  rounded-md
                  text-left
                  transition-colors duration-200
                  cursor-pointer
                  group
                "
                aria-label={`Search for ${query}`}
              >
                <p className="text-spotify-text-subdued text-base relative inline-block overflow-hidden">
                  <span className="absolute top-0 bottom-0 left-0 w-0 bg-[#8a05ff] group-hover:w-full transition-all duration-500 ease-out opacity-40 z-0"></span>
                  <span className="relative z-10">"{query}"</span>
                </p>
              </button>
            ))}
          </div>
        </div>
      </div>
      
      {/* CTA */}
    </div>
  );
};
