import { useState } from 'react';
import { SearchBar } from './components/SearchBar/SearchBar';
import { SongCard } from './components/SongCard/SongCard';
import { PlaylistHeader } from './components/PlaylistHeader/PlaylistHeader';
import { LoadingSpinner } from './components/LoadingSpinner/LoadingSpinner';
import { ErrorMessage } from './components/ErrorMessage/ErrorMessage';
import { EmptyState } from './components/EmptyState/EmptyState';
import { useRecommendations } from './hooks/useRecommendations';
import notePurple from './assets/note_purple.png';

/**
 * Main App Component
 * 
 * This is the root component of Sonic Mood.
 * It orchestrates the interaction between all child components and manages
 * the overall application state.
 * 
 * Architecture:
 * - Uses custom useRecommendations hook for API calls and state management
 * - Implements a clean component hierarchy with clear separation of concerns
 * - Provides responsive layout for mobile, tablet, and desktop
 * 
 * Features:
 * - Natural language music search
 * - Real-time loading states
 * - Error handling with retry functionality
 * - Empty state for first-time users
 * - Responsive design with Tailwind CSS
 * - Accessible markup following WCAG 2.1 guidelines
 */
function App() {
  // Custom hook for managing recommendations API calls
  const { 
    data, 
    error, 
    isLoading, 
    isIdle, 
    fetchRecommendations,
    reset 
  } = useRecommendations();
  
  // Store the current search query
  const [query, setQuery] = useState<string>('');
  
  // Store the last search query for retry functionality
  const [lastQuery, setLastQuery] = useState<string>('');
  
  /**
   * Handles search submission from the SearchBar component
   * Stores the query and calls the API to fetch recommendations
   * 
   * @param searchQuery - Natural language description of desired music
   */
  const handleSearch = async (searchQuery: string) => {
    setQuery(searchQuery);
    setLastQuery(searchQuery);
    await fetchRecommendations(searchQuery);
  };
  
  /**
   * Handles retry after an error
   * Re-submits the last query
   */
  const handleRetry = () => {
    if (lastQuery) {
      handleSearch(lastQuery);
    }
  };
  
  /**
   * Handles clicking the logo to return to the home page
   * Clears all state and returns to the initial empty state
   */
  const handleLogoClick = () => {
    setQuery('');
    setLastQuery('');
    reset();
  };
  
  return (
    <div className="min-h-screen bg-spotify-bg">
      {/* Header - Spotify style */}
      <header className="bg-spotify-elevated/95 backdrop-blur-lg sticky top-0 z-50 border-b border-white/5 shadow-lg">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <button 
              onClick={handleLogoClick}
              className="group flex items-center gap-3 cursor-pointer focus:outline-none rounded-lg px-2 py-1 -ml-2 relative z-10"
              aria-label="Return to home page"
            >
              <h1 className="text-2xl sm:text-3xl font-bold text-white group-hover:!text-[#8a05ff]">
                sonic mood
              </h1>
            </button>
            <div className="text-xs sm:text-sm text-spotify-text-subdued font-medium">
              Powered by pgvector
            </div>
          </div>
        </div>
      </header>
      
      {/* Main content */}
      <main className="container mx-auto px-4 py-12" role="main">
        {/* Purple note icon above search bar */}
        <div className="flex justify-center mb-8">
          <img 
            src={notePurple} 
            alt="" 
            className="w-32 h-32 transition-transform duration-300 hover:scale-110"
            aria-hidden="true"
          />
        </div>
        
        {/* Search bar - always visible at the top */}
        <div className="max-w-4xl mx-auto mb-16">
          <SearchBar 
            onSearch={handleSearch} 
            isLoading={isLoading}
            value={query}
            onChange={setQuery}
          />
        </div>
        
        {/* Content area - shows different states based on app state */}
        <div className="max-w-4xl mx-auto">
          {/* Loading state */}
          {isLoading && <LoadingSpinner />}
          
          {/* Error state */}
          {error && !isLoading && (
            <ErrorMessage error={error} onRetry={handleRetry} />
          )}
          
          {/* Success state - show recommendations */}
          {data && !isLoading && !error && (
            <div>
              {/* Playlist header with AI-generated name */}
              <PlaylistHeader playlistName={data.playlistName} />
              
              {/* List of song recommendations */}
              <div 
                className="space-y-5"
                role="list"
                aria-label="Song recommendations"
              >
                {data.recommendations.map((song, index) => (
                  <div 
                    key={song.id}
                    className={`animate-fade-in-up opacity-0 stagger-${index + 1}`}
                    style={{ animationFillMode: 'forwards' }}
                  >
                    <SongCard 
                      song={song} 
                      index={index}
                    />
                  </div>
                ))}
              </div>
              
              {/* Footer info */}
              <div className="mt-12 text-center">
                <div className="glass inline-block px-6 py-3 rounded-full">
                  <p className="text-gray-300 text-sm font-medium">
                    ✨ Powered by semantic search with <span className="text-primary-purple font-semibold">pgvector</span>
                  </p>
                </div>
              </div>
            </div>
          )}
          
          {/* Empty state - shown when no search has been performed */}
          {isIdle && <EmptyState onSearch={handleSearch} />}
        </div>
      </main>
      
      {/* Footer */}
      <footer className="bg-spotify-black mt-20">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-spotify-text-subdued text-center text-xs sm:text-sm">
            Built with React, TypeScript, and Tailwind CSS • 
            Deployed on{' '}
            <a 
              href="https://render.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-primary-purple hover:underline focus:outline-none focus:underline transition-colors"
            >
              Render
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
