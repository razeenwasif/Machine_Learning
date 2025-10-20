import React from 'react';
import Plot from 'react-plotly.js';
import { useTheme } from '@mui/material/styles';

interface Props {
  metrics: Record<string, number>;
}

const MetricsChart: React.FC<Props> = ({ metrics }) => {
  const theme = useTheme();
  const data = [
    {
      x: Object.keys(metrics),
      y: Object.values(metrics),
      type: 'bar' as const,
      marker: {
        color: theme.palette.primary.main,
        gradient: {
          type: 'vertical',
          color: [
            theme.palette.primary.light,
            theme.palette.primary.dark,
          ]
        }
      },
    },
  ];

  const layout = {
    title: 'Model Performance Metrics',
    paper_bgcolor: theme.palette.background.paper,
    plot_bgcolor: theme.palette.background.paper,
    font: {
      color: theme.palette.text.primary,
    },
    xaxis: {
      gridcolor: theme.palette.divider,
    },
    yaxis: {
      gridcolor: theme.palette.divider,
    },
  };

  return <Plot data={data} layout={layout} style={{ width: '100%' }} />;
};

export default MetricsChart;
