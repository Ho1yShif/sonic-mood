import { FiMusic } from 'react-icons/fi';

/**
 * Props for the PlaylistHeader component
 */
interface PlaylistHeaderProps {
  /** AI-generated playlist name to display */
  playlistName: string;
}

/**
 * PlaylistHeader Component - Spotify Style
 * 
 * Displays the AI-generated playlist name in a Spotify-inspired design.
 * Clean, minimal, and focused on the content.
 * 
 * @param props - PlaylistHeader props
 */
export const PlaylistHeader = ({ playlistName }: PlaylistHeaderProps) => {
  return (
    <div 
      className="mb-8 animate-fade-in"
      role="region"
      aria-label="Playlist information"
    >
      {/* Playlist info - Spotify style */}
      <div className="flex items-start gap-5">
        {/* Playlist "cover" */}
        <div 
          className="
            flex-shrink-0
            w-28 h-28
            bg-gradient-to-br from-primary-purple to-purple-600
            rounded-md
            flex items-center justify-center
            shadow-xl
          "
          aria-hidden="true"
        >
          <FiMusic className="text-5xl text-white" />
        </div>
        
        {/* Playlist details */}
        <div className="flex-1 self-end">
          <p className="text-xs font-bold text-white uppercase tracking-wider mb-2">
            Playlist
          </p>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-black text-white mb-3 leading-tight">
            {playlistName.toLowerCase()}
          </h2>
        </div>
      </div>
    </div>
  );
};
