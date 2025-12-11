# UniView Frontend

React + TypeScript + Vite application for UniView.

## Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **React Router** - Client-side routing
- **Context API** - State management

## Project Structure

```
frontend/
├── src/
│   ├── components/       # Reusable UI components
│   │   └── layout/       # Layout components (Navigation, etc.)
│   ├── context/          # React Context providers
│   ├── pages/            # Page components
│   ├── routes/           # Routing configuration
│   ├── styles/           # Global styles
│   ├── hooks/            # Custom React hooks
│   ├── services/         # API services
│   ├── types/            # TypeScript types
│   ├── utils/            # Utility functions
│   ├── App.tsx           # Root component
│   └── main.tsx          # Entry point
├── public/               # Static assets
└── index.html            # HTML template
```

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- npm or yarn

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

### Build

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Features

- ✅ Dark/Light mode toggle
- ✅ Responsive navigation
- ✅ Glass morphism UI
- ✅ Gradient text effects
- ✅ Smooth transitions

## Future Enhancements

- [ ] Authentication
- [ ] User profiles
- [ ] Course management
- [ ] Real-time notifications
