import React from 'react';
import { Box, Typography, Paper, Alert } from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import type { PipelineResult, RecordLinkageResult } from '../types';
import MetricsChart from './MetricsChart';
import RecordLinkageSummary from './RecordLinkageSummary';

interface Props {
  automlResult: PipelineResult | null;
  rlResult: RecordLinkageResult | null;
  automlError: string | null;
  rlError: string | null;
  automlLoading: boolean;
  rlLoading: boolean;
}

const ResultsDisplay: React.FC<Props> = ({ automlResult, rlResult, automlError, rlError, automlLoading, rlLoading }) => {
  const noResults = !automlResult && !rlResult;
  const noActivity = noResults && !automlLoading && !rlLoading;

  const animationProps = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 },
    transition: { duration: 0.5 },
  };

  return (
    <Paper elevation={3} sx={{ p: 2, height: '100%', overflow: 'auto' }}>
      <Typography variant="h5" gutterBottom>Results</Typography>

      {automlError && <Alert severity="error" sx={{ mb: 2 }}>{automlError}</Alert>}
      {rlError && <Alert severity="error" sx={{ mb: 2 }}>{rlError}</Alert>}

      {noActivity && (
        <Typography>Run a pipeline to see results here.</Typography>
      )}

      <AnimatePresence>
        {automlResult && (
          <motion.div {...animationProps}>
            <Box>
              <Typography variant="h6" gutterBottom>AutoML Result</Typography>
              <Typography variant="subtitle1">Model: {automlResult.model_name}</Typography>
              {automlResult.metrics && <MetricsChart metrics={automlResult.metrics} />}
            </Box>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {rlResult && (
          <motion.div {...animationProps}>
            <Box mt={4}>
              <RecordLinkageSummary result={rlResult} />
            </Box>
          </motion.div>
        )}
      </AnimatePresence>
    </Paper>
  );
};

export default ResultsDisplay;
