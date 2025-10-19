import { FaSpotify } from 'react-icons/fa';
import type { Song } from '../../types';

/**
 * Props for the SongCard component
 */
interface SongCardProps {
  /** Song data to display */
  song: Song;
  /** Index of the song in the list (for numbering) */
  index: number;
}

/**
 * SongCard Component
 * 
 * Displays a single song recommendation with:
 * - Song title (prominent)
 * - Artist name (secondary)
 * - Spotify link button (if available)
 * - Visual hover effects
 * - Accessible markup
 * 
 * The component gracefully handles missing Spotify links by showing
 * a disabled state instead of a clickable button.
 * 
 * @param props - SongCard props
 */
export const SongCard = ({ song, index }: SongCardProps) => {
  const { title, artist, spotifyLink } = song;
  
  return (
    <div 
      className="
        glass-card rounded-2xl p-6
        hover:ring-2 hover:ring-purple-500
        transition-shadow duration-300
        group
      "
      role="article"
      aria-label={`Song ${index + 1}: ${title} by ${artist}`}
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-5">
        {/* Song information */}
        <div className="flex items-start gap-5 flex-1 min-w-0">
          {/* Simple number indicator - Spotify style */}
          <div 
            className="
              flex-shrink-0
              w-6
              text-gray-400
              flex items-center justify-center
              font-normal text-base
            "
            aria-hidden="true"
          >
            {index + 1}
          </div>
          
          {/* Song details */}
          <div className="flex-1 min-w-0 self-center">
            <h3 className="text-xl sm:text-2xl font-bold text-white mb-1.5 truncate">
              {title}
            </h3>
            <p className="text-gray-300 text-base truncate font-medium">
              {artist}
            </p>
          </div>
        </div>
        
        {/* Spotify link button */}
        <div className="flex-shrink-0 sm:ml-4">
          {spotifyLink ? (
            <a
              href={spotifyLink}
              target="_blank"
              rel="noopener noreferrer"
              className="
                flex items-center gap-2
                px-5 py-3
                bg-black
                hover:bg-gray-900
                text-white
                rounded-xl
                font-semibold text-base
                focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 focus:ring-offset-black
                transition-all duration-200
                whitespace-nowrap
                border border-gray-800
                group/spotify
              "
              aria-label={`Listen to ${title} on Spotify`}
            >
              <FaSpotify className="text-[#1DB954] text-2xl" />
              <span className="text-sm text-white">Listen on <span className="font-bold text-white">Spotify</span></span>
            </a>
          ) : (
            <div
              className="
                flex items-center gap-2
                px-5 py-3
                bg-gray-800/50
                text-gray-500
                rounded-xl
                font-semibold text-base
                cursor-not-allowed
                border border-gray-700/50
                whitespace-nowrap
              "
              aria-label="Not available on Spotify"
            >
              <FaSpotify className="text-2xl opacity-50" />
              <span className="hidden sm:inline text-sm">Not available</span>
              <span className="sm:hidden text-sm">N/A</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
