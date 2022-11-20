package com.vc.deg.viz.om;

//import java.util.Arrays;
//import java.util.Random;

public class Solver {

//	// a[1..n][1..n] >= 0
//	static int[] hungarian1(float[][] a) {
//		int n = a.length - 1;
//
//		int[] u = new int[n + 1];
//		int[] v = new int[n + 1];
//		int[] p = new int[n + 1];
//		int[] way = new int[n + 1];
//		for (int i = 1; i <= n; ++i) {
//			p[0] = i;
//			int j0 = 0;
//			float[] minv = new float[n + 1];
//			Arrays.fill(minv, Float.MAX_VALUE);
//			boolean[] used = new boolean[n + 1];
//			do {
//				used[j0] = true;
//				int i0 = p[j0];
//				float delta = Float.MAX_VALUE;
//				int j1 = 0;
//				for (int j = 1; j <= n; ++j)
//					if (!used[j]) {
//						float cur = a[i0][j] - u[i0] - v[j];
//						if (cur < minv[j]) {
//							minv[j] = cur;
//							way[j] = j0;
//						}
//						if (minv[j] < delta) {
//							delta = minv[j];
//							j1 = j;
//						}
//					}
//				for (int j = 0; j <= n; ++j)
//					if (used[j]) {
//						u[p[j]] += delta;
//						v[j] -= delta;
//					} else
//						minv[j] -= delta;
//				j0 = j1;
//			} while (p[j0] != 0);
//			do {
//				int j1 = way[j0];
//				p[j0] = p[j1];
//				j0 = j1;
//			} while (j0 != 0);
//		}
//		return p;
//	}
//
//	static int[] hungarianF(float[][] dLut) {
//
//		int n = dLut.length;
//
//		float[][] a = new float [n+1][n+1];
//		for (int i = 1; i <= n; i++) 
//			for (int j = 1; j <= n; j++) 
//				a[i][j] = dLut[i-1][j-1];
//
//		int[] u = new int[n + 1];
//		int[] v = new int[n + 1];
//		int[] p = new int[n + 1];
//		int[] way = new int[n + 1];
//		for (int i = 1; i <= n; ++i) {
//			p[0] = i;
//			int j0 = 0;
//			float[] minv = new float[n + 1];
//			Arrays.fill(minv, Float.MAX_VALUE);
//			boolean[] used = new boolean[n + 1];
//			do {
//				used[j0] = true;
//				int i0 = p[j0];
//				float delta = Float.MAX_VALUE;
//				int j1 = 0;
//				for (int j = 1; j <= n; ++j)
//					if (!used[j]) {
//						float cur = a[i0][j] - u[i0] - v[j];
//						if (cur < minv[j]) {
//							minv[j] = cur;
//							way[j] = j0;
//						}
//						if (minv[j] < delta) {
//							delta = minv[j];
//							j1 = j;
//						}
//					}
//				for (int j = 0; j <= n; ++j)
//					if (used[j]) {
//						u[p[j]] += delta;
//						v[j] -= delta;
//					} else
//						minv[j] -= delta;
//				j0 = j1;
//			} while (p[j0] != 0);
//			do {
//				int j1 = way[j0];
//				p[j0] = p[j1];
//				j0 = j1;
//			} while (j0 != 0);
//		}
//
//		int[] perm = new int[n];
//
//		//float err = 0;
//		for (int i = 0; i < n; i++) {
//			perm[i] = p[i+1]-1;
//			//err += dLut[perm[i]][i];
//		}
//		//System.out.println("H cost:" + err);
//
//		return perm;
//	}
//	
//	static int[] hungarian(int[][] dLut) {
//
//		int n = dLut.length;
//
//		int[][] a = new int [n+1][n+1];
//		for (int i = 1; i <= n; i++) 
//			for (int j = 1; j <= n; j++) 
//				a[i][j] = dLut[i-1][j-1];
//
//		int[] u = new int[n + 1];
//		int[] v = new int[n + 1];
//		int[] p = new int[n + 1];
//		int[] way = new int[n + 1];
//		for (int i = 1; i <= n; ++i) {
//			p[0] = i;
//			int j0 = 0;
//			int[] minv = new int[n + 1];
//			Arrays.fill(minv, Integer.MAX_VALUE);
//			boolean[] used = new boolean[n + 1];
//			do {
//				used[j0] = true;
//				int i0 = p[j0];
//				int delta = Integer.MAX_VALUE;
//				int j1 = 0;
//				for (int j = 1; j <= n; ++j)
//					if (!used[j]) {
//						int cur = a[i0][j] - u[i0] - v[j];
//						if (cur < minv[j]) {
//							minv[j] = cur;
//							way[j] = j0;
//						}
//						if (minv[j] < delta) {
//							delta = minv[j];
//							j1 = j;
//						}
//					}
//				for (int j = 0; j <= n; ++j)
//					if (used[j]) {
//						u[p[j]] += delta;
//						v[j] -= delta;
//					} else
//						minv[j] -= delta;
//				j0 = j1;
//			} while (p[j0] != 0);
//			do {
//				int j1 = way[j0];
//				p[j0] = p[j1];
//				j0 = j1;
//			} while (j0 != 0);
//		}
//
//		int[] perm = new int[n];
//
//		int err = 0;
//		for (int i = 0; i < n; i++) {
//			perm[i] = p[i+1]-1;
//			//err += dLut[perm[i]][i];
//		}
//
//		//System.out.println("HI cost:" + err);
//
//		return perm;
//	}
//
//	static int[] pathGrowing1(float[][] distLut) {
//
//		int n = distLut.length;
//		int[] m0 = new int[n];
//
//		int set = 0;
//		int x = 0;
//
//		for (int i = 0; i < 2*n-1; i++) {
//
//			float min = Float.MAX_VALUE;
//			int minY = -1;
//			if (set == 1) {
//				for (int y = 0; y < n; y++) {
//					if (distLut[x][y] < min) {
//						min = distLut[x][y];
//						minY = y;
//					}
//					distLut[x][y] += 100000;
//				}	
//			}
//			else {
//				for (int y = 0; y < n; y++) {
//					if (distLut[y][x] < min) {
//						min = distLut[y][x];
//						minY = y;
//					}
//					distLut[y][x] += 100000;
//				}
//				m0[x] = minY;
//			}
//			set = 2-(set+1);
//			x = minY;
//		}
//
//		return m0;
//	}
//
//	static int[] pathGrowing2(int[][] distLut) {
//
//		int n = distLut.length;
//		int[] m0 = new int[n];
//		int[] m1 = new int[n];
//		Arrays.fill(m1, -1);
//
//		int set = 0;
//		int x = 0;
//		int sum0 = 0, sum1 = 0;
//
//		int[] indices = new int[n];
//
//		for (int i = 0; i < 2*n-1; i++) {
//
//			float min = Float.MAX_VALUE;
//			int minY = -1;
//			if (set == 1) {
//				for (int y = 0; y < n; y++) {
//					if (distLut[x][y] < min) {
//						min = distLut[x][y];
//						minY = y;
//					}
//					distLut[x][y] += 100000;
//				}	
//				m1[minY] = x;
//				indices[x] = -1;
//				sum1 += min;
//			}
//			else {
//				for (int y = 0; y < n; y++) {
//					if (distLut[y][x] < min) {
//						min = distLut[y][x];
//						minY = y;
//					}
//					distLut[y][x] += 100000;
//				}
//				m0[x] = minY;
//				sum0 += min;
//			}
//			set = 2-(set+1);
//			x = minY;
//		}
//
//		for (int i = 0; i < indices.length; i++) {
//			if (indices[i] != -1) {
//				x = i;
//				break;
//			}
//		}
//
//		int y = -1;
//		for (int i = 0; i < m1.length; i++) {
//			if (m1[i] == -1) {
//				y = i;
//				break;
//			}
//		}
//		sum1 += distLut[y][x] - 100000;
//		m1[y] = x;
//
//		System.out.println("PG cost:      " + Math.min(sum0, sum1));
//		
//		return (sum0 < sum1) ? m0 : m1;		
//	}
//	
//	static int[] greedy(int[][] matrix) {
//		int dim = matrix.length;
//		
//		int[] selected = new int[dim];
//		int[] assignment = new int[dim];
//		for (int i = 0; i < assignment.length; i++) 
//			assignment[i] = -1;
//		
//		int cost = 0;
//		for (int k = 0; k < dim; k++) {
//			int minCost = Integer.MAX_VALUE;
//			int selectedJob = -1;
//			int is =  -1;
//
//			for (int i = 0; i < dim; i++) {
//				if ( selected[i] == 0) {
//					for (int j = 0; j < dim; j++) {
//						if (assignment[j] == -1 && matrix[i][j] < minCost ) {
//							selectedJob = j;
//							minCost = matrix[i][j];
//							is = i;
//						}
//					}	
//				}
//			}
//			cost += minCost;
//			selected[is] = 1;
//			assignment[selectedJob] = is;
//		}
//		System.out.println("Greedy cost:  " + cost);
//		return assignment;
//	}

	public static int[] lapjv(int[][] matrix) {
		int i, imin, i0, freerow;
		int j, j1, j2=0, endofpath=0, last=0, min=0;

		int dim = matrix.length;
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

//		int lapcost = 0;
//		for (i = 0; i < dim; i++) {
//			j = inRow[i];
//			//u[i] = matrix[i][j] - v[j];
//			lapcost += matrix[i][j];
//		}
//		System.out.println("VF cost:      " + lapcost);
		
		return inCol;
	}
	
//	static int[] lapjvF(float[][] matrix) {
//		boolean unassignedfound;
//		int i, imin, numfree=0, prvnumfree, i0, freerow;
//		int j, j1, j2=0, endofpath=0, last=0, low, up;
//		float v2, usubmin, h, umin, min=0;
//		
//		int dim = matrix.length;
//		int[] inRow = new int[dim];
//		int[] inCol = new int[dim];
//		//float[] u = new float[dim]; 
//		float[] v = new float[dim]; 
//		int[] free = new int[dim]; 
//		int[] collist = new int[dim]; 
//		int[] matches = new int[dim]; 
//		int[] pred = new int[dim]; 
//		
//		float[] d = new float[dim]; 
//		
//		// skipping L53-54
//		for (j = dim - 1; j >= 0; j--) {
//			min = matrix[0][j];
//			imin = 0;
//			for (i = 1; i < dim; i++) {
//				matrix[i][j] = matrix[i][j];
//				if (matrix[i][j] < min) {
//					min = matrix[i][j];
//					imin = i;
//				}
//			}
//
//			v[j] = min;
//			matches[imin]++;
//			if (matches[imin] == 1) {
//				inRow[imin] = j;
//				inCol[j] = imin;
//			} else {
//				inCol[j] = -1;
//			}
//		}
//
//		for (i = 0; i < dim; i++) {
//			if (matches[i] == 0) {
//				free[numfree] = i;
//				numfree++;
//			} else if (matches[i] == 1) {
//				j1 = inRow[i];
//				min = Float.MAX_VALUE;
//				for (j = 0; j < dim; j++) {
//					if (j != j1 && matrix[i][j]-v[j] < min) {
//						min = matrix[i][j] - v[j];
//					}
//				}
//				v[j1] -= min;
//			}
//		}
//
//		for (int loopcmt = 0; loopcmt < 2; loopcmt++) {
//			int k = 0;
//			prvnumfree = numfree;
//			numfree = 0;
//			while (k < prvnumfree) {
//				i = free[k];
//				k++;
//				umin = matrix[i][0] - v[0];
//				j1 = 0;
//				usubmin = Float.MAX_VALUE;
//
//				for (j = 1; j < dim; j++) {
//					h = matrix[i][j] - v[j];
//
//					if (h < usubmin) {
//						if (h >= umin) {
//							usubmin = h;
//							j2 = j;
//						} else {
//							usubmin = umin;
//							umin = h;
//							j2 = j1;
//							j1 = j;
//						}
//					}
//				}
//
//				i0 = inCol[j1];
//				if (umin < usubmin) {
//					v[j1] = v[j1] - (usubmin - umin);
//				} else if (i0 >= 0) {
//					j1 = j2;
//					i0 = inCol[j2];
//				}
//
//				inRow[i] = j1;
//				inCol[j1] = i;
//				if (i0 >= 0) {
//					if (umin < usubmin) {
//						k--;
//						free[k] = i0;
//					} else {
//						free[numfree] = i0;
//						numfree++;
//					}
//				}
//			}
//		}
//
//		for (int f = 0; f < numfree; f++) {
//			freerow = free[f];
//			for (j = 0; j < dim; j++) {
//				d[j] = matrix[freerow][j] - v[j];
//				pred[j] = freerow;
//				collist[j] = j;
//			}
//
//			low = 0;
//			up = 0;
//			unassignedfound = false;
//
//			while (!unassignedfound) {
//				if (up == low) {
//					last = low - 1;
//					min = d[collist[up]];
//					up++;
//
//					for (int k = up; k < dim; k++) {
//						j = collist[k];
//						h = d[j];
//						if (h <= min) {
//							if (h < min) {
//								up = low;
//								min = h;
//							}
//							collist[k] = collist[up];
//							collist[up] = j;
//							up++;
//						}
//					}
//
//					for (int k = low; k < up; k++) {
//						if (inCol[collist[k]] < 0) {
//							endofpath = collist[k];
//							unassignedfound = true;
//							break;
//						}
//					}
//				}
//
//				if (!unassignedfound) {
//					j1 = collist[low];
//					low++;
//					i = inCol[j1];
//					h = matrix[i][j1] - v[j1] - min;
//
//					for (int k = up; k < dim; k++) {
//						j = collist[k];
//						v2 = matrix[i][j] - v[j] - h;
//
//						if (v2 < d[j]) {
//							pred[j] = i;
//
//							if (v2 == min) {
//								if (inCol[j] < 0) {
//									endofpath = j;
//									unassignedfound = true;
//									break;
//								} else {
//									collist[k] = collist[up];
//									collist[up] = j;
//									up++;
//								}
//							}
//
//							d[j] = v2;
//						}
//					}
//				}
//			}
//
//			for (int k = 0; k <= last; k++) {
//				j1 = collist[k];
//				v[j1] += d[j1] - min;
//			}
//
//			i = freerow + 1;
//			while (i != freerow) {
//				i = pred[endofpath];
//				inCol[endofpath] = i;
//				j1 = endofpath;
//				endofpath = inRow[i];
//				inRow[i] = j1;
//			}
//		}
//
////		float lapcost = 0;
////		for (i = 0; i < dim; i++) {
////			j = inRow[i];
////			//u[i] = matrix[i][j] - v[j];
////			lapcost += matrix[i][j];
////		}
////		System.out.println("LF cost: " + lapcost);
//		
//		return inCol;
//	}

	

//	public static void main(String[] args) {
//
//		int dim = 40;
//		float[][] a = new float[dim][dim]; 
////		= {
////				{ 177, 203, 192, 242, 183}, 
////				{ 153, 157, 208, 175, 159}, 
////				{ 174, 270, 112, 287, 187}, 
////				{ 224, 145, 386, 137, 199}, 
////				{ 163, 147, 232, 183, 162}
////		};
//		
//		int[][] ai = 
////				new int[dim][dim]; 
//		 {
//				{ 177, 203, 192, 242, 183}, 
//				{ 153, 157, 208, 175, 159}, 
//				{ 174, 270, 112, 287, 187}, 
//				{ 224, 145, 386, 137, 199}, 
//				{ 163, 147, 232, 183, 162}
//		};
//		
////		Random random = new Random();
////
////		for (int i = 0; i < a.length; i++) {
////			for (int j = 0; j < a[0].length; j++) {
////				a[i][j] = (float)(1000*random.nextFloat());
////				ai[i][j] = (int)a[i][j];
////			}
////		}
//
//		//int[] p1 = pathGrowing2(a);	
//
//		long t0 = System.nanoTime();
//		int[] p0 = greedy(ai);
//		long t1 = System.nanoTime();
//		System.out.println("greedy time:   " + ((t1-t0))/1000);
//		//System.out.println(Arrays.toString(p0));
//		System.out.println();
//		
//			
////		long t0 = System.nanoTime();
////		int[] p0 = hungarianF(a);
////		System.out.println();
////		long t1 = System.nanoTime();
////		System.out.println("hungarian time: " + ((t1-t0))/1000);
////		System.out.println(Arrays.toString(p0));
////		System.out.println();
//		
//		
//		int[] p2 = lapjv(ai);
//		long t2 = System.nanoTime();
//		System.out.println("VF time:       " + ((t2-t1))/1000);
//		//System.out.println(Arrays.toString(p2));	
//		
//		
//		System.out.println();
//		int[] p1 = pathGrowing2(ai);
//		long t3 = System.nanoTime();
//		
//		System.out.println("pathGrow time: " + ((t3-t2))/1000);
//		//System.out.println(Arrays.toString(p1));
//	}
	
}
