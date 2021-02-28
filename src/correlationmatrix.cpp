/* This code is designed to solve min 0.5*<X-G, X-G> s.t. X_ii =1, i=1,2,...,n		*/
/* and X>=tau*I (symmetric and positive semi-definite)								*/
/* based on the algorithm  in "A Quadratically Convergent Newton Method for			*/
/* Computing the Nearest Correlation Matrix											*/
/* By Houduo Qi and Defeng Sun														*/
/* SIAM J. Matrix Anal. Appl. 28 (2006) 360--385.									*/

/* This particular C implementation is a result of the summer research project		*/
/* of Pawel Zaczkowski, Imperial College London,									*/
/* with Professor Defeng Sun, National University of Singapore						*/

/* Last modified date:  August 11, 2010												*/
/* The  input argument is the given symmetric G										*/
/* The outputs are the optimal primal and dual solutions							*/
/* Diagonal Preconditioner is added													*/

/* Please send your comments and suggestions to										*/
/* pawelzaczkowski@cantab.net or matsundf@nus.edu.sg								*/

/* Warning: Accuracy may not be guaranteed!!!!!										*/

/* Xiangjie Xue tries to translate into Eigen world, no attempt is made to further improve the performance */

#include <RcppEigen.h>

#define PERTURBATION 1.0e-9

// void Jacobian_matrix (double* x, struct matrix* Omega12, struct matrix* P, struct matrix* Ax)
void Jacobian_matrix (Eigen::VectorXd& x, Eigen::MatrixXd& Omega12, Eigen::MatrixXd& P, Eigen::VectorXd& Ax)
{
	// int n = P->rows;
	int n = P.rows();
	// int r=Omega12->rows;
	int r = Omega12.rows();
	// int s=Omega12->columns;
	int s = Omega12.cols();

	// for (i=0; i<n; i++) Ax->entries[i] = 0;
	Ax.setZero();
	
	// struct matrix H1;
	// H1.rows = n;
	// H1.columns = r;
	// H1.entries = (double *) malloc (sizeof(double) * n*r);
	Eigen::MatrixXd H1(n, r);
	
	// struct matrix temp;
	Eigen::MatrixXd temp;
	
	// struct matrix Omega;
	// Omega.rows = r;
	// Omega.columns = s;
	// Omega.entries = (double*) malloc(sizeof(double) * r * s);
	Eigen::MatrixXd Omega(r, s);
	
	if (r>0)
	{
		if (r< ((double) n)/2.0)
		{
			// temp.rows = r;
			// temp.columns = s;
			// temp.entries = (double *) malloc(sizeof(double) * r *s);
			temp.resize(r, s);
			
			// for(i=0; i<n; i++)
			// 	for(j=0; j<r; j++)
			// 		H1.entries[j*n + i] = x[i] * P->entries[j*n + i];  //H1 = diag(x) * P1
			H1 = x.head(n).asDiagonal() * P;
			
			//Omega12 = Omega12.*(H1'*P(:,r+1:n))
			// cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, r, s, n, 1, H1.entries, n, P->entries+n*r, n, 0, temp.entries, r);
			temp = H1.block(0, 0, n, r).transpose() * P.block(0, r, n, s);
			// for (i=0; i<r*s; i++) Omega.entries[i] =Omega12->entries[i] * temp.entries[i];
			Omega12.array() *= temp.array();
			
			// free(temp.entries);
			
			// struct matrix HT;
			// HT.rows = n;
			// HT.columns = n;
			// HT.entries = (double *) malloc(sizeof(double) * n*n);
			// for (i=0; i<n*n; i++) HT.entries[i] = 0; //make sure it's 0s
			Eigen::MatrixXd HT(n, n);
			HT.setZero();
			
			// P1^T * H1
			// temp.rows = r;
			// temp.columns =r; 
			// temp.entries = (double*) malloc(sizeof(double) * r * r);
			temp.resize(r, r);
			// cblas_dgemm(CblasColMajor, CblasTrans ,CblasNoTrans, r, r, n, 1, P->entries, n, H1.entries, n, 0, temp.entries,r);
			temp = P.block(0, 0, n, r).transpose() * H1.block(0, 0, n, r);
			
			//HT += P1 * P1^T * H1
			// cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, r, r, 1, P->entries, n, temp.entries, r, 0, HT.entries, n);
			HT.block(0, 0, n, r) = P.block(0, 0, n, r) * temp;
			
			// free(temp.entries);
			
			//HT = P1 * P1^T * H1 + P2 * Omega^T
			// cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, r, s, 1, P->entries+n*r, n, Omega.entries, r, 1, HT.entries, n);
			HT.block(0, 0, n, r) += P.block(0, r, n, s) * Omega.block(0, 0, r, s).transpose();
			
			//HT = P1 * P1^T * H1 + P2 * Omega^T ; P1 * Omega^T
			// cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, s, r, 1, P->entries, n, Omega.entries, r, 0, HT.entries+n*r, n);
			HT.block(0, r, n, s) = P.block(0, 0, n, r) * Omega.block(0, 0, r, s);
			
			
			// for (i=0; i<n; i++)
			// {
			// 	for (j=0; j<n; j++)
			// 		Ax->entries[i] += P->entries[j*n+i]*HT.entries[j*n+i];
			// 	Ax->entries[i] += PERTURBATION*x[i];
			// }
			Ax.head(n) = P.cwiseProduct(HT).rowwise().sum() + PERTURBATION * x;
			// free(HT.entries);
			// free(H1.entries);
		}	
		else 
		{
			// if(r==n) for (i=0; i<n; i++) Ax->entries[i] = x[i] * (1.+PERTURBATION);
			if(r==n) Ax.head(n) = x * (1.+PERTURBATION);
			else 
			{				
				// H1.rows = n;
				// H1.columns = s;
				// free(H1.entries);
				// H1.entries = (double *) malloc (sizeof(double) * n *s);
				Eigen::MatrixXd H1(n, s);
				
				// for(i=0; i<n; i++)
				// 	for(j=0; j<s; j++)
				// 		H1.entries[j*n + i] = x[i] * P->entries[(j+r)*n + i];  //H1 = diag(x) * P2
				H1 = x.head(n).asDiagonal() * P.block(0, r, n, s);
				
				//Omega12 = ones(r,s)-Omega12;
				// for(i=0; i<r; i++) for (j=0; j<s; j++) Omega.entries[j*r+i] = 1. - Omega12->entries[j*r+i];
				Omega12.block(0, 0, r, s).array() = 1. -  Omega12.block(0, 0, r, s).array();
				
				
				// temp.rows = r;
				// temp.columns = s;
				// temp.entries = (double *) malloc(sizeof(double) * r *s);
				temp.resize(r, s);
				
				//Omega12 = Omega12.*((P(:,1:r))'*H2);
				// cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, r, s, n, 1, P->entries, n, H1.entries, n, 0, temp.entries, r);
				temp = P.block(0, 0, n, r).transpose() * H1.block(0, 0, n, s);
				// for (i=0; i<r*s; i++) Omega.entries[i] *= temp.entries[i];
				Omega.block(0, 0, r, s) += temp;
				// free(temp.entries);
				
				// struct matrix HT;
				// HT.rows = n;
				// HT.columns = n;
				// HT.entries = (double *) malloc(sizeof(double) * n*n);
				// for (i=0; i<n*n; i++) HT.entries[i] = 0; //make sure it's 0s
				Eigen::MatrixXd HT(n, n);
				HT.setZero();
				
				//HT += P2 * Omega^T
				// cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, r, s, 1, P->entries+n*r, n, Omega.entries, r, 0, HT.entries, n);
				HT.block(0, 0, n, r) = P.block(0, r, n, s) * Omega.block(0, 0, r, s).transpose();
				
				// cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, s, s, n, 1, H1.entries, n, P->entries+n*r, n,0, temp.entries, s);
				temp.block(0, 0, s, s) = H1.block(0, 0, n, s).transpose() * P.block(0, r, n, s);
				
				// cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, s, s, 1, P->entries+n*r, n, temp.entries, s,0, HT.entries+r*n, n);
				HT.block(0, r, n, s) = P.block(0, r, n, s) * temp.block(0, 0, s, s);
				
				// cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, s, r, 1, P->entries, n, Omega.entries, r, 1, HT.entries+r*n, n);
				HT.block(0, r, n, s) += P.block(0, 0, n, r) * Omega.block(0, 0, r, s);
				
				// for (i=0; i<n; i++)
				// {
				// 	Ax->entries[i] = 0;
				// 	for (j=0; j<n; j++)
				// 		Ax->entries[i] -= P->entries[j*n+i]*HT.entries[j*n+i];
				// 	Ax->entries[i] += x[i] *(1. + PERTURBATION);
				// }
				Ax.head(n) = P.cwiseProduct(HT).rowwise().sum() * -1. + x.head(n) * (1. + PERTURBATION);
				// free(HT.entries);
				// free(H1.entries);
			}
		}
	}
	// free(Omega.entries);
}

// void pre_cg(double* b,double tol,int maxit,double* c,struct matrix* Omega12,struct matrix* P, double* p, int* flag, double* relres, int* iterk)
void pre_cg(Eigen::VectorXd & b,double & tol,int & maxit, Eigen::VectorXd & c,
	Eigen::MatrixXd & Omega12, Eigen::MatrixXd & P, Eigen::VectorXd & p, int& flag, double& relres, int& iterk)
{
	// int n = P->rows;
	int n = P.rows();
	// double* r = (double*) malloc (sizeof(double) *n);
	// for(i=0; i<n; i++) r[i] = b[i];
	Eigen::VectorXd r(b);
	
	// double n2b = norm(b, n);
	double n2b = b.norm();
	
	double tolb = tol * n2b;
	
	// for(i=0; i<n; i++) p[i] = 0;
	p.setZero();
	
	flag = 1;
	iterk = 0;
	relres = 1000;
	
	// double* z = (double*) malloc(sizeof(double) * n);
	// for(i=0; i<n; i++) z[i] = r[i] / c[i];
	Eigen::VectorXd z = r.cwiseQuotient(c.head(n));
	
	// double rz1 = 0;
	// for(i=0; i<n; i++) rz1 += r[i]*z[i];
	double rz1 = r.dot(z);

	double rz2 = 1;
	
	// double *d = (double*) malloc(sizeof(double) * n);
	// for(i=0; i<n; i++) d[i] = z[i];
	Eigen::VectorXd d(z);
	
	int k;
	
	// struct matrix w;
	// w.rows = n;
	// w.columns = 1;
	// w.entries = (double*) malloc(sizeof(double)*n);
	Eigen::VectorXd w(n);
	
	for(k=1; k<=maxit; k++)
	{
		// if(k>1) for(i=0; i<n; i++) d[i] = z[i] + d[i]*rz1/rz2;
		if (k > 1) d = z + d * (rz1 / rz2);
		
		Jacobian_matrix(d, Omega12, P, w);
		
		// double denom = 0;
		// for (i=0; i<n; i++) denom += d[i] * w.entries[i];
		double denom = d.dot(w);
		
		iterk = k;
		
		// double normr = norm(r, n);
		double normr = r.norm();
		
		relres = normr / n2b;
		
		if(denom<=0)
		{
			// double normd = norm(d, n);
			// for(i=0; i<n; i++) p[i] = d[i] / normd;
			p = d / d.norm();
			break;
		}
		else 
		{
			double alpha = rz1/denom;
			// for(i=0; i<n; i++) 
			// {
			// 	p[i] = p[i] + alpha * d[i];
			// 	r[i] = r[i] - alpha * w.entries[i];
			// }
			p += alpha * d;
			r -= alpha * w;
		}
		
		// for(i=0; i<n; i++) z[i] = r[i] / c[i];
		z = r.cwiseQuotient(c.head(n));
		
		// normr = norm(r, n);
		normr = r.norm();
		
		if(normr <= tolb)
		{
			iterk = k;
			relres = normr / n2b;
			flag = 0;
			break;
		}
		rz2 = rz1;
		
		// rz1 = 0;
		// for(i=0; i<n; i++) rz1 += r[i] * z[i];
		rz1 = r.dot(z);
	}
	// free(w.entries);
	// free(r);
	// free(z);
	// free(d);
}

// void omega_mat(double* lambda, int n, struct matrix * Omega12) //assume lambda is already sorted in descending order
void omega_mat(Eigen::VectorXd& lambda, int & n, Eigen::MatrixXd & Omega12) //assume lambda is already sorted in descending order
{
	int r=0;
	while(lambda[r]>0 && r<n) r++;
	
	if (r==0) 
	{
		// Omega12->rows = 0;
		// Omega12->columns = 0;
		// Omega12->entries = NULL;
		Omega12.resize(0, 0);
	}
	else if (r == n)
	{
		// int i=0;
		// Omega12->rows = n;
		// Omega12->columns = n;
		// Omega12->entries = (double*) malloc (sizeof(double)*n*n);
		// for (i=0; i<n*n; i++) Omega12->entries[i] = 1;
		Omega12.resize(n, n);
		Omega12.setOnes();
	}
	else
	{
		// int i,j;
		// Omega12->rows = r;
		// Omega12->columns = n-r;
		// Omega12->entries = (double*) malloc(sizeof(double)*r*(n-r));
		// for(j=0; j<n-r; j++)
		// 	for(i=0; i<r; i++)
		// 		Omega12->entries[j*r+i] = lambda[i] / (lambda[i] - lambda[r+j]);
		Omega12.resize(r, n - r);
		Omega12 = lambda.replicate(1, n - r).cwiseQuotient(lambda.replicate(1, n - r).rowwise() - lambda.tail(n - r).transpose());
	}
}

// void precond_matrix (struct matrix* Omega12, struct matrix * P, double* c)
void precond_matrix (Eigen::MatrixXd& Omega12, Eigen::MatrixXd& P, Eigen::VectorXd& c)
{
	// int r = Omega12->rows;
	int r = Omega12.rows();
	// int s = Omega12->columns;
	int s = Omega12.cols();
	// int n = P->rows;
	int n = P.rows();
	int i,j;
	
	// for(i=0; i<n; i++) c[i] = 1.;
	c.setOnes();
	
	if(r==n) return;
	
	//H=P'; H=H.*H;
	// struct matrix H;
	// H.rows = n;
	// H.columns = n;
	// H.entries = (double*) malloc(sizeof(double) *n *n);
	// for(i=0; i<n; i++) for(j=0; j<n; j++) H.entries[j*n + i ] = P->entries[j*n+i]*P->entries[j*n+i];
	Eigen::MatrixXd H = P.block(0, 0, n, n).cwiseAbs2();
	
	if(r>0)
	{
		if((double) r < (double) n /2.)
		{
			// struct matrix H12;
			// H12.rows = n;
			// H12.columns = s;
			// H12.entries = (double*) malloc (sizeof(double) * n *s);
			Eigen::MatrixXd H12(n, s);
			
			//H12 = H(1:r,:)'*Omega12;
			// cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, s, r, 1, H.entries, n, Omega12->entries, r, 0, H12.entries, n);
			H12 = H.block(0, 0, n, r) * Omega12.block(0, 0, r, s);		
			
			// for (i=0; i<n; i++)
			// {
			// 	// c(i) = sum(H(1:r,i))*(d'*H(1:r,i));
			// 	c[i] = 0;
			// 	// for(j=0; j<r; j++) c[i] += H.entries[i+j*n];
			// 	for(j=0; j<r; j++) c[i] += H(i, j);
			// 	c[i] *= c[i];
				
			// 	//c(i) = c(i) +2.0*(H12(i,:)*H(r+1:n,i));
			// 	// for(j=0; j<s; j++) c[i] += 2.0 * H12.entries[i+j*n]*H.entries[i+(r+j)*n];
			// 	for(j=0; j<s; j++) c[i] += 2.0 * H12(i, j) * H(i, (r+j));
				
			// 	// if(c[i] < 1.0e-8) c[i] = 1.0e-8;
				
			// }
			c = H.block(0, 0, n, r).rowwise().sum().cwiseAbs2().cwiseMax(1e-8);
			// free(H12.entries);
		}
		else 
		{
			// struct matrix H12;
			// H12.rows = r;
			// H12.columns = n;
			// H12.entries = (double*) malloc (sizeof(double) * r *n);
			Eigen::MatrixXd H12(r, n);
			
			// struct matrix Omega; 
			// Omega.rows = Omega12->rows;
			// Omega.columns = Omega12->columns;
			// Omega.entries = (double*) malloc(sizeof(double) * Omega.rows * Omega.columns);
			Eigen::MatrixXd Omega(Omega12.rows(), Omega12.cols());
			
			// for(i=0; i<r; i++) for(j=0; j<s; j++) Omega.entries[i+j*r] = 1. - Omega12->entries[i+j*r];
			Omega.block(0, 0, r, s).array() = 1. - Omega12.block(0, 0, r, s).array();
			
			//H12 = Omega12*H(r+1:n,:);
			// cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, r, n, s, 1, Omega.entries, r, H.entries+r*n, n, 0, H12.entries, r);	
			H12 = Omega.block(0, 0, r, s) * H.block(0, r, n, s).transpose();
			
			for (i=0; i<n; i++)
			{
				// c(i) = sum(H(r+1:n,i))*(d'*H(r+1:n,i));
				double temp = 0;
				c[i]=0;
				// for(j=0; j<s; j++) c[i] += H.entries[i+(j+r)*n];
				for(j=0; j<s; j++) c[i] += H(i, (j+r));
				c[i] *= c[i];
				
				// c(i) = c(i) + 2.0*(H(1:r,i)'*H12(:,i));
				// for(j=0; j<r; j++) c[i] += 2.0 * H.entries[i+n*j]*H12.entries[i*r+j];
				for(j=0; j<r; j++) c[i] += 2.0 * H(i, j) * H12(j, i);
				
				// double alpha = 0;
				// for (j=0; j<n; j++) alpha += H.entries[i+j*n]; 
				double alpha = H.row(i).sum();
				
				// temp =0;
				// for(j=0; j<n; j++) temp += H.entries[i+j*n];
				temp = alpha;
				
				c[i] = -c[i] + alpha*temp;
				
				// if(c[i] < 1.0e-8) c[i] = 1.0e-8;
			}
			c = c.cwiseMax(1e-8);
			// free (H12.entries);
			// free(Omega.entries);
		}
		
	}
	// free(H.entries);
}

// void gradient(double *y, double* lambda, struct matrix* P, double* b0, double* f, double* Fy) 
void gradient(Eigen::VectorXd &y , Eigen::VectorXd &lambda, Eigen::MatrixXd & P, Eigen::VectorXd & b0, double & f, Eigen::VectorXd & Fy) 
{
	// *f = 0;
	f = 0;
	// int n = P->rows;
	int n = P.rows();
	
	// struct matrix Q;
	// Q.rows = P->rows;
	// Q.columns = P->columns;
	// Q.entries = (double*) malloc(sizeof(double) * Q.rows * Q.columns);
	Eigen::MatrixXd Q(P.rows(), P.cols());
	
	// for (i=0; i<n; i++) for (j=0; j<n; j++) Q.entries[j+i*n] = P->entries[j+i*n] * sqrt(max(lambda[i], 0));
	Q = P.cwiseProduct(lambda.head(n).cwiseMax(0).cwiseSqrt().transpose().replicate(P.rows(), 1));
	// for (i=0; i<n; i++) 
	// {
	// 	Fy[i] = 0;
	// 	for (j=0; j<n; j++) Fy[i] += Q.entries[j*n+i]*Q.entries[i+j*n];
	// }
	Fy.head(n) = Q.rowwise().squaredNorm();
	// for (i=0; i<n; i++) *f += pow(max(lambda[i], 0), 2.);
	f = lambda.head(n).cwiseMax(0).squaredNorm();
	// double temp=0;
	// for (i=0; i<n; i++) temp+= b0[i]*y[i];
	double temp = b0.head(n).dot(y.head(n));
	f = f/2. - temp;
	// free(Q.entries);
}

// void Correlation_Newton(struct matrix* G, struct matrix* X, double* y)
void Correlation_Newton(Eigen::MatrixXd & G, Eigen::MatrixXd & X, Eigen::VectorXd & y)
{
	// printf("Newton method starts...\n");
	// int i,j;
	// int n = G->rows;
	int n = G.rows();
	
	double tau = 0;
	// double t0 =(double) clock()/(double) CLOCKS_PER_SEC;
	
	// double* b; 
	// b = (double*) malloc(sizeof(double) * n);
	// for (i=0; i<n; i++) b[i] = 1;
	Eigen::VectorXd b(n);
	b.setOnes();
	
	// double *b0;
	// b0 = (double*) malloc (sizeof(double) * n);
	// for (i=0; i<n; i++) b0[i] = b[i];
	Eigen::VectorXd b0(b);
	
	//make G symmetric
	// for (i=0; i<n; i++) for(j=i+1; j<n; j++)
	// {
	// 	G->entries[j*n+i] = (G->entries[j*n+i]+G->entries[j+i*n])/2;
	// 	G->entries[j+i*n] = G->entries[j*n+i];
	// }
	G = (G + G.transpose()) / 2.;
	
	// for(i=0; i<n; i++) G->entries[i*n+i] = G->entries[i*n+i] - tau;
	G.diagonal().array() -= tau;
	// for(i=0; i<n; i++) b0[i] = b0[i] - tau;
	b0.array() -= tau;
	
	// double Res_b[300];
	// for(i=0; i<300; i++) Res_b[i] = 0;
	Eigen::VectorXd Res_b(300);
	Res_b.setZero();
	
	//initial point
	// for(i=0; i<n; i++) y[i] = 0;
	y.setZero();
	
	// double* Fy; 
	// Fy = (double *) malloc (sizeof(double) * n);
	Eigen::VectorXd Fy(n);
	
	// for(i=0; i<n; i++) Fy[i] = 0;
	Fy.setZero();
	
	int k=0;
	int f_eval = 0;
	
	int Iter_Whole = 200;
	int iter_Inner = 20;
	int maxit = 200;
	int iterk = 0;
	double tol = 1.0e-2;
	
	double error_tol = 1.0e-6;
	double sigma_1 = 1.0e-4;
	
	// double * x0;
	// x0 = (double *) malloc(sizeof(double) *n);
	// for (i=0; i<n; i++) x0[i] = y[i];
	Eigen::VectorXd x0(y.head(n));
	
	// double prec_time = 0;
	// double pcg_time = 0;
	// double eig_time = 0;
	
	// double *c = (double *) malloc (sizeof(double) *n);
	// for (i=0; i<n; i++) c[i] =1 ;
	Eigen::VectorXd c(n);
	c.setOnes();
	
	// double *d = (double*) malloc(sizeof(double) *n );
	// for(i=0; i<n; i++) d[i] = 0;
	Eigen::VectorXd d(n);
	d.setZero();
	
	// double val_G = 0;
	// for (i=0; i<n; i++) for(j=0; j<n; j++) val_G += G(i, j)*G(i, j)/2.;
	// double val_G = G.squaredNorm() / 2.;
	
	// X->entries = (double*) malloc(sizeof(double) *n *n);
	// // X = G + diag(y)
	// for(i=0; i<n; i++) for(j=0; j<n; j++)
	// {
	// 	if(i==j) X->entries[n*j+i] = G->entries[n*j+i] + y[i];
	// 	else X->entries[n*j+i] = G->entries[n*j+i];
	// }
	X.resize(n, n);
	X = G;
	X.diagonal() += y.head(n);
	
	// double eig_time0 = (double) clock()/(double) CLOCKS_PER_SEC;
	
	// double *lambda = (double*) malloc(sizeof(double) * n);
	Eigen::VectorXd lambda(n);
	// int info;
	// char jobz = 'V';
	// char uplo = 'L';
	// int lwork = 1+6*n+2*n*n;
	// int liwork = 3+5*n;
	// //copy G as evec decomposition kills it..
	// struct matrix P;
	// P.rows = n;
	// P.columns = n;
	// P.entries = (double*) malloc (sizeof(double) * n * n);
	// for (j=0; j<n; j++) for (i=0; i<n; i++) P.entries[j*n+i] = X->entries[j*n+i];
	// double* a = (double*) malloc(sizeof(double) * lwork);
	// int* aa = (int*) malloc(sizeof(int) * liwork);
	
	//G contains the evecs now, lambda contains evals; if info = 0, evals are in ascending order
	// dsyevd_(&jobz, &uplo, &G->rows, P.entries, &G->rows, lambda, a, &lwork, aa, &liwork, &info);
	//dsyev_(&jobz, &uplo, &G->rows, P.entries, &G->rows, lambda, a, &lwork, &info);
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig;
	eig.compute(X);
	Eigen::MatrixXd P = eig.eigenvectors();
	lambda = eig.eigenvalues();
	
	// eig_time = eig_time + (double) clock()/(double) CLOCKS_PER_SEC - eig_time0;
	
	/* we want the evals in descending order; dsyev returns them in ascending order at best. 
	 If that's the case, we simply revert the order.
	 In case they are scattered, we would need to sort it together with matrix P attached to it. This is a bit messy, tbd.. */
	// if (info == 0)
	// {
	// 	for(j=0; j<n/2; j++)
	// 	{
	// 		if(lambda[j] < lambda[n-1-j])
	// 		{
	// 			double tmp = lambda[j];
	// 			lambda[j] = lambda[n-1-j];
	// 			lambda[n-1-j] = tmp;
	// 			for(i=0; i<n; i++)
	// 			{
	// 				tmp = P.entries[n*j + i];
	// 				P.entries[n*j+i] = P.entries[n*(n-1-j)+i];
	// 				P.entries[n*(n-1-j)+i] = tmp;
	// 			}
	// 		}
	// 	}
	// }
	// else 
	// {
	// 	printf("Evals not in ascending order!!!\n");
	// 	return;
	// }
	P.rowwise().reverseInPlace();
	lambda.reverseInPlace();
	
	double f0;
	gradient(y, lambda, P, b0, f0, Fy);
	
	double f = f0;

	f_eval = f_eval + 1;
	// for(i=0; i<n; i++) b[i] = b0[i] - Fy[i];
	b = b0 - Fy;
	// double norm_b = norm(b, n);
	double norm_b = b.norm();
	
	// double Initial_f = val_G - f0;
	
	// printf("Newton: Initial Dual Objective Function value ==== %13.6f \n", Initial_f);
	// printf("Newton: Norm of Gradient:        %13.6f \n",norm_b);
	
	// struct matrix Omega12;
	Eigen::MatrixXd Omega12;
	omega_mat(lambda, n, Omega12);
	// for (i=0; i<n; i++) x0[i] = y[i];
	x0 = y;
	
	while(norm_b>error_tol && k< Iter_Whole)
	{
		// Rcpp::Rcout<<X<<std::endl;
		// double prec_time0 = (double) clock()/(double) CLOCKS_PER_SEC;
		
		precond_matrix(Omega12, P, c);
		
		// prec_time = prec_time + (double) clock()/(double) CLOCKS_PER_SEC - prec_time0;
		
		// double pcg_time0 = (double) clock()/(double) CLOCKS_PER_SEC;
		
		int flag; 
		double relres;
		
		pre_cg(b, tol, maxit, c, Omega12, P, d, flag, relres, iterk);
		
		// pcg_time = pcg_time + (double) clock()/(double) CLOCKS_PER_SEC - pcg_time0;
		
		// printf("Newton: Number of CG Iterations: %13d \n", iterk);

		// if (flag!=0) printf("..... Not a full Newton step......, flag = %d ", flag);
		
		// double slope = 0;
		// for(i=0; i<n; i++) slope = slope + (Fy[i] - b0[i])*d[i];
		double slope = d.dot(Fy - b0);
		
		// for (i=0; i<n; i++) y[i] = x0[i] + d[i];
		y = x0 + d;
		
		// X = G + diag(y)
		// for(i=0; i<n; i++) for(j=0; j<n; j++)
		// {
		// 	if(i==j) X->entries[n*j+i] = G->entries[n*j+i] + y[i];
		// 	else X->entries[n*j+i] = G->entries[n*j+i];
		// }
		X = G;
		X.diagonal() += y.head(n);
		
		// eig_time0 = (double) clock()/(double) CLOCKS_PER_SEC;
		
		// for (j=0; j<n; j++) for (i=0; i<n; i++) P.entries[j*n+i] = X->entries[j*n+i];
		P = X;
			
		//G contains the evecs now, lambda contains evals; if info = 0, evals are in ascending order
		// dsyevd_(&jobz, &uplo, &G->rows, P.entries, &G->rows, lambda, a, &lwork, aa, &liwork, &info);
		//dsyev_(&jobz, &uplo, &G->rows, P.entries, &G->rows, lambda, a, &lwork, &info);
		eig.compute(P);
		P = eig.eigenvectors();
		lambda = eig.eigenvalues();
		
		// eig_time = eig_time + (double) clock()/(double) CLOCKS_PER_SEC - eig_time0;
		
		// if (info == 0)
		// {
		// 	for(j=0; j<n/2+1; j++)
		// 	{
		// 		if(lambda[j] < lambda[n-1-j])
		// 		{
		// 			double tmp = lambda[j];
		// 			lambda[j] = lambda[n-1-j];
		// 			lambda[n-1-j] = tmp;
		// 			for(i=0; i<n; i++)
		// 			{
		// 				tmp = P.entries[n*j + i];
		// 				P.entries[n*j+i] = P.entries[n*(n-1-j)+i];
		// 				P.entries[n*(n-1-j)+i] = tmp;
		// 			}
		// 		}
		// 	}
		// }
		// else 
		// {
		// 	printf("Evals not in ascending order!!!\n");
		// 	return;
		// }
		P.rowwise().reverseInPlace();
		lambda.reverseInPlace();
		
		gradient(y, lambda, P, b0, f, Fy);
		
		int k_inner = 0;
		
		while(k_inner<=iter_Inner && f>f0 + sigma_1*pow(0.5, k_inner) * slope	+ 10e-6)
		{
			k_inner++;
			// for (i=0; i<n; i++) y[i] = x0[i] + pow(.5, k_inner) * d[i];
			y = x0 + std::pow(0.5, k_inner) * d;
			
			// for(i=0; i<n; i++) for(j=0; j<n; j++)
			// {
			// 	if(i==j) X->entries[n*j+i] = G->entries[n*j+i] + y[i];
			// 	else X->entries[n*j+i] = G->entries[n*j+i];
			// }
			X = G;
			X.diagonal() += y.head(n);
			
			// eig_time0 = (double) clock()/(double) CLOCKS_PER_SEC;
			
			// for (j=0; j<n; j++) for (i=0; i<n; i++) P.entries[j*n+i] = X->entries[j*n+i];
			P = X;
				
			//G contains the evecs now, D contains evals; if info = 0, evals are in ascending order
			// dsyevd_(&jobz, &uplo, &G->rows, P.entries, &G->rows, lambda, a, &lwork, aa, &liwork, &info);
			//dsyev_(&jobz, &uplo, &G->rows, P.entries, &G->rows, lambda, a, &lwork, &info);
			eig.compute(P);
			P = eig.eigenvectors();
			lambda = eig.eigenvalues();
			
			// eig_time = eig_time + (double) clock()/(double) CLOCKS_PER_SEC - eig_time0;
			
			// if (info == 0)
			// {
			// 	for(j=0; j<n/2+1; j++)
			// 	{
			// 		if(lambda[j] < lambda[n-1-j])
			// 		{
			// 			double tmp = lambda[j];
			// 			lambda[j] = lambda[n-1-j];
			// 			lambda[n-1-j] = tmp;
			// 			for(i=0; i<n; i++)
			// 			{
			// 				tmp = P.entries[n*j + i];
			// 				P.entries[n*j+i] = P.entries[n*(n-1-j)+i];
			// 				P.entries[n*(n-1-j)+i] = tmp;
			// 			}
			// 		}
			// 	}
			// }
			// else 
			// {
			// 	printf("Evals not in ascending order!!!\n");
			// 	return;
			// }
			P.rowwise().reverseInPlace();
			lambda.reverseInPlace();

			gradient(y, lambda, P, b0, f, Fy);			
		}
		
		f_eval = f_eval + k_inner + 1;
		
		// for(i=0; i<n; i++) x0[i] = y[i];
		x0 = y;
		f0 = f;
		
		k++;
		
		// for(i=0; i<n; i++) b[i] = b0[i] - Fy[i];
		b = b0 - Fy;
		
		// norm_b = norm(b,n);
		norm_b = b.norm();
		
		// printf("Newton: Norm of Gradient:        %13.6f \n",norm_b);
		
		Res_b[k] = norm_b;
		
		// if(Omega12.rows!=0) free(Omega12.entries);
		omega_mat(lambda, n, Omega12);		
	}
	
	int r = 0;
	while(lambda[r]>0 && r<n) r++;
	
	// if (r==0) for(i=0; i<n*n; i++) X->entries[i] = 0;
	if (r==0) X.setZero();
	else if (r==n);
	else 
	{
		if((double) r<=(double) n /2.)
		{
			if(r>1)
			{
				// for (j=0; j<r; j++)	for(i=0; i<n; i++) P.entries[i+j*n] *= sqrt(lambda[j]);
				P.block(0, 0, n, r) = P.block(0, 0, n, r).cwiseProduct(lambda.head(r).cwiseSqrt().transpose().replicate(n, 1));
				// cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, r, 1, P.entries, n, P.entries, n, 0, X->entries, n);
				X.block(0, 0, n, n) = P.block(0, 0, n, r) * P.block(0, 0, n, r).transpose();
			}
			else
			{
				// cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, r, lambda[0]*lambda[0], P.entries, n, P.entries, n, 0, X->entries, n);
				X.block(0, 0, n, n) = P.block(0, 0, n, r) * P.block(0, 0, n, r).transpose() * (lambda[0] * lambda[0]);
			}
		}
		else //r>n/2
		{
			// for (j=0; j<n-r; j++) for(i=0; i<n; i++) P.entries[i+(j+r)*n] *= sqrt(-lambda[r+j]);
			P.block(0, r, n, n - r) = P.block(0, r, n, n - r).cwiseProduct((-1. * lambda.tail(n - r)).cwiseSqrt().transpose().replicate(n, 1));
			// cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n-r, 1, P.entries+n*r, n, P.entries+n*r, n, 1, X->entries, n);
			X.block(0, 0, n, n) += P.block(0, r, n, n - r) * P.block(0, r, n, n - r).transpose();
		}
	}

	// free(b);
	// free(b0);
	// free(Fy);
	// free(c);
	// free(d);
	// free(lambda);
	// free(P.entries);
	// free(a);
	// free(aa);
	
	// double Final_f = val_G - f;
	// double val_obj = 0;
	// for(i=0; i<n*n; i++) val_obj += pow(X->entries[i] - G->entries[i], 2)/2;
	// double val_obj = (X - G).squaredNorm() / 2.;
	
	// for(i=0; i<n; i++) X->entries[i+n*i] += tau;
	X.diagonal().array() += tau;
	
	// double time_used = (double) clock()/(double) CLOCKS_PER_SEC - t0;
	
	// printf("\n");
	
	// printf("Newton: Number of Iterations ============================================= %13d \n", k);
	// printf("Newton: Number of Function Evaluations =================================== %13d \n", f_eval);
	// printf("Newton: Final Dual Objective Function value ============================== %13.6f \n",Final_f);
	// printf("Newton: Final Original Objective Function value ========================== %13.6f \n", val_obj);
	// printf("Newton: The rank of the Optimal Solution ================================= %13d \n",r);
	
	// printf("Newton: computing time for computing preconditioners ===================== %13.6f \n", prec_time);
	// printf("Newton: computing time for linear systems solving (cgs time) ============= %13.6f \n", pcg_time);
	// printf("Newton: computing time for  eigenvalue decompostions (calling eig time)=== %13.6f \n", eig_time);
	// printf("Newton: computing time used for equal weight calibration ================= %13.6f \n",time_used);
}

// int main(int argc, const char * argv[])
// {
// 	int i;
// 	int N = 3000;
// 	struct matrix G;
// 	G.rows = N;
// 	G.columns = N;
// 	G.entries = (double*) malloc(sizeof(double) * N * N);
// 	srand((int) time(NULL));
// 	for (i=0; i<N*N; i++) G.entries[i] = 2.* (double) rand()/ (double) RAND_MAX-1.;
// 	struct matrix X;
// 	X.rows = N;
// 	X.columns = N;
// 	double y[N];
	
// 	Correlation_Newton(&G, &X, y);
	
// 	//printMatrix(&X);
	
// 	return 0;
// }

// [[Rcpp::export]]
Eigen::MatrixXd CorrelationMatrixcpp(const Eigen::MatrixXd &G1){
	Eigen::MatrixXd G(G1), ans(G1.rows(), G1.cols());
	Eigen::VectorXd y(G1.rows());
	Correlation_Newton(G, ans, y);
	return ans;
}