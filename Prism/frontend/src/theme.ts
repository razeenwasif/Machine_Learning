import { createTheme } from '@mui/material/styles';

// A custom theme for this app
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9', // A lighter, more vibrant blue
    },
    secondary: {
      main: '#f48fb1', // A vibrant pink for secondary actions
    },
    background: {
      default: '#121212', // This will be overridden by the global CSS gradient
      paper: 'rgba(30, 30, 30, 0.7)', // Semi-transparent paper for a modern feel
    },
    text: {
      primary: '#e0e0e0',
      secondary: '#b0bec5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 700,
      color: '#ffffff',
    },
    h5: {
      fontWeight: 500,
      color: '#f5f5f5',
    },
    h6: {
      color: '#e0e0e0'
    }
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none', // Ensure paper components don't have gradients from MUI
          backdropFilter: 'blur(10px)', // Frosted glass effect
          borderColor: 'rgba(255, 255, 255, 0.12)',
          borderWidth: '1px',
          borderStyle: 'solid',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 'bold',
          borderRadius: '8px',
          padding: '10px 20px',
          transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
          '&:hover': {
            transform: 'scale(1.05)',
            boxShadow: '0 0 20px rgba(144, 202, 249, 0.5)',
          },
        },
        containedPrimary: {
          background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
          color: 'white',
        },
        containedSecondary: {
          background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)',
          color: 'white',
        },
      },
    },
  },
});

export default theme;
