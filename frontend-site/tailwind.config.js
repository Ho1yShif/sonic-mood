/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        'sans': ['Neue Montreal', 'sans-serif'],
      },
      colors: {
        'primary-purple': '#8a05ff',
        'primary-black': '#000000',
        'primary-white': '#ffffff',
        // Spotify-inspired dark theme
        'spotify-black': '#000000',
        'spotify-bg': '#121212',
        'spotify-elevated': '#1a1a1a',
        'spotify-card': '#181818',
        'spotify-card-hover': '#282828',
        'spotify-text': '#ffffff',
        'spotify-text-subdued': '#b3b3b3',
      },
    },
  },
  plugins: [],
}

