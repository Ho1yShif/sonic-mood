import type { RecommendationResponse } from '../types';

/**
 * Mock data for Sonic Mood
 * These are realistic sample responses that match the expected API format
 */

/**
 * Mock response for workout/energetic music queries
 */
export const mockWorkoutResponse: RecommendationResponse = {
  playlistName: "Energetic Workout Vibes",
  recommendations: [
    {
      id: "1",
      title: "Eye of the Tiger",
      artist: "Survivor",
      spotifyLink: "https://open.spotify.com/track/2KH16WveTQWT6KOG9Rg6e2"
    },
    {
      id: "2",
      title: "Stronger",
      artist: "Kanye West",
      spotifyLink: "https://open.spotify.com/track/4fzsfWzRhPawzqhX8Qt9F3"
    },
    {
      id: "3",
      title: "Till I Collapse",
      artist: "Eminem",
      spotifyLink: null // Example of missing Spotify link
    },
    {
      id: "4",
      title: "Thunderstruck",
      artist: "AC/DC",
      spotifyLink: "https://open.spotify.com/track/57bgtoPSgt236HzfBOd8kj"
    },
    {
      id: "5",
      title: "Lose Yourself",
      artist: "Eminem",
      spotifyLink: "https://open.spotify.com/track/5Z01UMMf7V1o0MzF86s6WJ"
    }
  ]
};

/**
 * Mock response for relaxing/chill music queries
 */
export const mockRelaxingResponse: RecommendationResponse = {
  playlistName: "Chill Vibes for Focus",
  recommendations: [
    {
      id: "6",
      title: "Weightless",
      artist: "Marconi Union",
      spotifyLink: "https://open.spotify.com/track/3jjujdWJ72nww5eGnfs2E7"
    },
    {
      id: "7",
      title: "So What",
      artist: "Miles Davis",
      spotifyLink: "https://open.spotify.com/track/77RLEJPKKgmXlkYcoOJqXJ"
    },
    {
      id: "8",
      title: "Avril 14th",
      artist: "Aphex Twin",
      spotifyLink: "https://open.spotify.com/track/3EWmNzMoz5zBF8VzSyC1oI"
    },
    {
      id: "9",
      title: "Claire de Lune",
      artist: "Claude Debussy",
      spotifyLink: null
    },
    {
      id: "10",
      title: "Weightless",
      artist: "Marconi Union",
      spotifyLink: "https://open.spotify.com/track/3jjujdWJ72nww5eGnfs2E7"
    }
  ]
};

/**
 * Mock response for sad/emotional music queries
 */
export const mockSadResponse: RecommendationResponse = {
  playlistName: "Melancholy Indie Heartbreak",
  recommendations: [
    {
      id: "11",
      title: "The Night We Met",
      artist: "Lord Huron",
      spotifyLink: "https://open.spotify.com/track/7nd9d15tgKXYAIgJsqf6rC"
    },
    {
      id: "12",
      title: "Skinny Love",
      artist: "Bon Iver",
      spotifyLink: "https://open.spotify.com/track/5wYb9fObRX4JBlMXf5wcLK"
    },
    {
      id: "13",
      title: "Someone Like You",
      artist: "Adele",
      spotifyLink: "https://open.spotify.com/track/1zwMYTA5nlNjZxYrvBB2pV"
    },
    {
      id: "14",
      title: "Mad World",
      artist: "Gary Jules",
      spotifyLink: null
    },
    {
      id: "15",
      title: "Hurt",
      artist: "Johnny Cash",
      spotifyLink: "https://open.spotify.com/track/1EV4FbEGMCSPRHMj3zjDcC"
    }
  ]
};

/**
 * Mock response for upbeat/happy music queries
 */
export const mockHappyResponse: RecommendationResponse = {
  playlistName: "Feel-Good Summer Hits",
  recommendations: [
    {
      id: "16",
      title: "Happy",
      artist: "Pharrell Williams",
      spotifyLink: "https://open.spotify.com/track/60nZcImufyMA1MKQY3dcCH"
    },
    {
      id: "17",
      title: "Walking on Sunshine",
      artist: "Katrina and the Waves",
      spotifyLink: "https://open.spotify.com/track/05wIrZSwuaVWhcv5FfqeH0"
    },
    {
      id: "18",
      title: "Don't Stop Me Now",
      artist: "Queen",
      spotifyLink: "https://open.spotify.com/track/7hQJA50XrCWABAu5v6QZ4i"
    },
    {
      id: "19",
      title: "Good Vibrations",
      artist: "The Beach Boys",
      spotifyLink: "https://open.spotify.com/track/6S7gOsEGrD2dZXq4yzFbPo"
    },
    {
      id: "20",
      title: "Mr. Blue Sky",
      artist: "Electric Light Orchestra",
      spotifyLink: null
    }
  ]
};

/**
 * Mock response for party/dance music queries
 */
export const mockPartyResponse: RecommendationResponse = {
  playlistName: "Dance Floor Bangers",
  recommendations: [
    {
      id: "21",
      title: "Uptown Funk",
      artist: "Mark Ronson ft. Bruno Mars",
      spotifyLink: "https://open.spotify.com/track/32OlwWuMpZ6b0aN2RZOeMS"
    },
    {
      id: "22",
      title: "Can't Stop the Feeling!",
      artist: "Justin Timberlake",
      spotifyLink: "https://open.spotify.com/track/3BKD4pXLU4a6vVCcqSQDGW"
    },
    {
      id: "23",
      title: "Shut Up and Dance",
      artist: "Walk the Moon",
      spotifyLink: "https://open.spotify.com/track/4kbj5MwxO1bq9wjT5g9HaA"
    },
    {
      id: "24",
      title: "September",
      artist: "Earth, Wind & Fire",
      spotifyLink: "https://open.spotify.com/track/5W3cjX2J3tjhG8zb6u0qHn"
    },
    {
      id: "25",
      title: "Get Lucky",
      artist: "Daft Punk ft. Pharrell Williams",
      spotifyLink: "https://open.spotify.com/track/2Foc5Q5nqNiosCNqttzHof"
    }
  ]
};

/**
 * Default mock response for generic queries
 */
export const mockDefaultResponse: RecommendationResponse = {
  playlistName: "Eclectic Mix for Every Mood",
  recommendations: [
    {
      id: "26",
      title: "Bohemian Rhapsody",
      artist: "Queen",
      spotifyLink: "https://open.spotify.com/track/7tFiyTwD0nx5a1eklYtX2J"
    },
    {
      id: "27",
      title: "Billie Jean",
      artist: "Michael Jackson",
      spotifyLink: "https://open.spotify.com/track/5ChkMS8OtdzJeqyybCc9R5"
    },
    {
      id: "28",
      title: "Hotel California",
      artist: "Eagles",
      spotifyLink: null
    },
    {
      id: "29",
      title: "Smells Like Teen Spirit",
      artist: "Nirvana",
      spotifyLink: "https://open.spotify.com/track/4CeeEOM32jQcH3eN9Q2dGj"
    },
    {
      id: "30",
      title: "Hey Jude",
      artist: "The Beatles",
      spotifyLink: "https://open.spotify.com/track/0aym2LBJBk9DAYuHHutrIl"
    }
  ]
};

/**
 * Array of all mock responses for variety
 */
export const allMockResponses = [
  mockWorkoutResponse,
  mockRelaxingResponse,
  mockSadResponse,
  mockHappyResponse,
  mockPartyResponse,
  mockDefaultResponse
];

