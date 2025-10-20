import React from 'react';
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableRow,
  Paper,
  Grid,
} from '@mui/material';
import type { RecordLinkageResult } from '../types';

interface Props {
  result: RecordLinkageResult;
}

const MetricTable: React.FC<{ title: string; data: Record<string, number> }> = ({ title, data }) => (
  <Box mb={3}>
    <Typography variant="subtitle1" gutterBottom>{title}</Typography>
    <TableContainer component={Paper} variant="outlined">
      <Table size="small">
        <TableBody>
          {Object.entries(data).map(([key, value]) => (
            <TableRow key={key}>
              <TableCell component="th" scope="row">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</TableCell>
              <TableCell align="right">{Number.isInteger(value) ? value : value.toFixed(4)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  </Box>
);

const RecordLinkageSummary: React.FC<Props> = ({ result }) => {
  return (
    <Box>
      <Typography variant="h6" gutterBottom>Record Linkage Summary</Typography>
      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
        {result.match_count} matches found for dataset key: {result.dataset_key}
      </Typography>
      <Grid container spacing={2} mt={1}>
        <Grid xs={12} md={4}>
          <MetricTable title="Linkage Quality" data={result.linkage_metrics} />
        </Grid>
        <Grid xs={12} md={4}>
          <MetricTable title="Blocking Summary" data={result.blocking_metrics} />
        </Grid>
        <Grid xs={12} md={4}>
          <MetricTable title="Runtime (seconds)" data={result.runtime} />
        </Grid>
      </Grid>
    </Box>
  );
};

export default RecordLinkageSummary;
