package com.vc.deg.viz.om;


public class JonkerVolgenantSolver {

	/**
	 * https://github.com/heetch/lapjv/blob/master/solver.go
	 * 
	 * @param matrix
	 * @return
	 */
	public static int[] computeAssignment(int[][] matrix) {
		return computeAssignment(matrix, matrix.length);
	}

	public static int[] computeAssignment(int[][] matrix, int dim) {
		int i, imin, i0, freerow;
		int j, j1, j2=0, endofpath=0, last=0, min=0;

		int[] inRow = new int[dim];
		int[] inCol = new int[dim];

		//int[] u = new int[dim]; 
		int[] v = new int[dim]; 
		int[] free = new int[dim]; 
		int[] collist = new int[dim]; 
		int[] matches = new int[dim]; 
		int[] pred = new int[dim]; 

		int[] d = new int[dim]; 

		// skipping L53-54
		for (j = dim - 1; j >= 0; j--) {
			min = matrix[0][j];
			imin = 0;
			for (i = 1; i < dim; i++) {
				if (matrix[i][j] < min) {
					min = matrix[i][j];
					imin = i;
				}
			}

			v[j] = min;
			matches[imin]++;
			if (matches[imin] == 1) {
				inRow[imin] = j;
				inCol[j] = imin;
			} else {
				inCol[j] = -1;
			}
		}

		int numfree=0;
		for (i = 0; i < dim; i++) {
			if (matches[i] == 0) {
				free[numfree] = i;
				numfree++;
			} else if (matches[i] == 1) {
				j1 = inRow[i];
				min = Integer.MAX_VALUE;
				for (j = 0; j < dim; j++) {
					if (j != j1 && matrix[i][j]-v[j] < min) {
						min = matrix[i][j] - v[j];
					}
				}
				v[j1] -= min;
			}
		}

		for (int loopcmt = 0; loopcmt < 2; loopcmt++) {
			int k = 0;
			int prvnumfree = numfree;
			numfree = 0;
			while (k < prvnumfree) {
				i = free[k];
				k++;
				int umin = matrix[i][0] - v[0];
				j1 = 0;
				int usubmin = Integer.MAX_VALUE;

				for (j = 1; j < dim; j++) {
					int h = matrix[i][j] - v[j];

					if (h < usubmin) {
						if (h >= umin) {
							usubmin = h;
							j2 = j;
						} else {
							usubmin = umin;
							umin = h;
							j2 = j1;
							j1 = j;
						}
					}
				}

				i0 = inCol[j1];
				if (umin < usubmin) {
					v[j1] = v[j1] - (usubmin - umin);
				} else if (i0 >= 0) {
					j1 = j2;
					i0 = inCol[j2];
				}

				inRow[i] = j1;
				inCol[j1] = i;
				if (i0 >= 0) {
					if (umin < usubmin) {
						k--;
						free[k] = i0;
					} else {
						free[numfree] = i0;
						numfree++;
					}
				}
			}
		}

		for (int f = 0; f < numfree; f++) {
			freerow = free[f];
			for (j = 0; j < dim; j++) {
				d[j] = matrix[freerow][j] - v[j];
				pred[j] = freerow;
				collist[j] = j;
			}

			int low = 0;
			int up = 0;
			boolean unassignedfound = false;

			while (!unassignedfound) {
				if (up == low) {
					last = low - 1;
					min = d[collist[up]];
					up++;

					for (int k = up; k < dim; k++) {
						j = collist[k];
						int h = d[j];
						if (h <= min) {
							if (h < min) {
								up = low;
								min = h;
							}
							collist[k] = collist[up];
							collist[up] = j;
							up++;
						}
					}

					for (int k = low; k < up; k++) {
						if (inCol[collist[k]] < 0) {
							endofpath = collist[k];
							unassignedfound = true;
							break;
						}
					}
				}

				if (!unassignedfound) {
					j1 = collist[low];
					low++;
					i = inCol[j1];
					int h = matrix[i][j1] - v[j1] - min;

					for (int k = up; k < dim; k++) {
						j = collist[k];
						int v2 = matrix[i][j] - v[j] - h;

						if (v2 < d[j]) {
							pred[j] = i;

							if (v2 == min) {
								if (inCol[j] < 0) {
									endofpath = j;
									unassignedfound = true;
									break;
								} else {
									collist[k] = collist[up];
									collist[up] = j;
									up++;
								}
							}

							d[j] = v2;
						}
					}
				}
			}

			for (int k = 0; k <= last; k++) {
				j1 = collist[k];
				v[j1] += d[j1] - min;
			}

			i = freerow + 1;
			while (i != freerow) {
				i = pred[endofpath];
				inCol[endofpath] = i;
				j1 = endofpath;
				endofpath = inRow[i];
				inRow[i] = j1;
			}
		}

		return inRow;
	}
}