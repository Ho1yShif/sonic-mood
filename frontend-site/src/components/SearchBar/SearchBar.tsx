import { useState, useRef, useEffect } from 'react';
import type { FormEvent, ChangeEvent } from 'react';
import { FiSearch } from 'react-icons/fi';

/**
 * Props for the SearchBar component
 */
interface SearchBarProps {
  /** Callback function called when the user submits a search */
  onSearch: (query: string) => void;
  /** Whether a search is currently in progress */
  isLoading?: boolean;
  /** Placeholder text for the input field */
  placeholder?: string;
  /** Optional controlled value for the input */
  value?: string;
  /** Optional callback when input value changes */
  onChange?: (value: string) => void;
}

/**
 * SearchBar Component
 * 
 * A clean, standard search input field for natural language music queries.
 * Features:
 * - Clean, minimal design with purple accents
 * - Single-line input with inline button
 * - Submit button with loading state
 * - Keyboard support (Enter to submit)
 * - Focus states with purple accent
 * 
 * @param props - SearchBar props
 */
export const SearchBar = ({ 
  onSearch, 
  isLoading = false,
  placeholder = "Search for music... (e.g., 'upbeat workout songs' or 'chill acoustic vibes')",
  value: controlledValue,
  onChange: controlledOnChange
}: SearchBarProps) => {
  const [internalQuery, setInternalQuery] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const MAX_CHARS = 500;
  
  // Use controlled value if provided, otherwise use internal state
  const query = controlledValue !== undefined ? controlledValue : internalQuery;
  const setQuery = controlledOnChange !== undefined ? controlledOnChange : setInternalQuery;
  
  // Detect OS for keyboard shortcut display
  const [isMac, setIsMac] = useState(true);
  
  useEffect(() => {
    // Detect if user is on Mac
    setIsMac(navigator.platform.toUpperCase().indexOf('MAC') >= 0);
    
    // Add keyboard shortcut listener for cmd+k / ctrl+k
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    
    document.addEventListener('keydown', handleKeyDown);
    
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);
  
  /**
   * Handles form submission
   * Prevents default form behavior and calls onSearch if query is valid
   */
  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    
    // Don't submit if query is empty or only whitespace
    if (query.trim()) {
      onSearch(query.trim());
    }
  };
  
  /**
   * Handles input change with character limit enforcement
   */
  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    
    // Enforce character limit
    if (value.length <= MAX_CHARS) {
      setQuery(value);
    }
  };
  
  // Calculate characters remaining
  const charsRemaining = MAX_CHARS - query.length;
  const isNearLimit = charsRemaining < 50;
  
  return (
    <div className="w-full max-w-4xl mx-auto">
      <form onSubmit={handleSubmit} className="w-full">
        {/* Main search container */}
        <div className="relative flex items-center">
          {/* Search icon */}
          <div className="absolute left-4 pointer-events-none">
            <FiSearch className="text-spotify-text-subdued text-xl" />
          </div>
          
          {/* Search input */}
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={handleChange}
            placeholder={placeholder}
            disabled={isLoading}
            className="
              w-full pl-12 pr-24 py-4
              bg-white/5 text-white placeholder-spotify-text-subdued
              border-2 border-white/10
              rounded-full
              focus:outline-none focus:border-primary-purple focus:bg-white/8
              hover:border-white/20 hover:bg-white/8
              disabled:opacity-50 disabled:cursor-not-allowed
              text-base
              transition-all duration-200
            "
            aria-label="Search for music (Press Cmd+K or Ctrl+K to focus)"
            maxLength={MAX_CHARS}
          />
          
          {/* Keyboard shortcut indicator */}
          <div className="absolute right-9 pointer-events-none hidden md:flex items-center gap-1 text-spotify-text-subdued text-sm">
            <span>{isMac ? '⌘' : 'Ctrl'}</span>
            <span>K</span>
          </div>
        </div>
        
        {/* Character counter (only show when approaching limit) */}
        {isNearLimit && (
          <div className="mt-2 text-right">
            <span 
              className={`
                text-xs font-medium
                ${charsRemaining < 20 ? 'text-red-400' : 'text-spotify-text-subdued'}
              `}
              aria-live="polite"
            >
              {charsRemaining} characters remaining
            </span>
          </div>
        )}
      </form>
    </div>
  );
};

