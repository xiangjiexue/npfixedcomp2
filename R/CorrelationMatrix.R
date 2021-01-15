
#################                                                                                              #################
#################  This R code is designed by Ying Cui at National University of Singapore to solve            #################
#################                               min  0.5*<X-G, X-G>                                            #################
#################                               s.t. X_ii =b_i, i=1,2,...,n                                    #################
#################                                    X>=tau*I (symmetric and positive semi-definite)           #################
#################                      #############################################                           #################
#################                            based on the algorithm  in                                        #################
#################  ''A Quadratically Convergent Newton Method for Computing the Nearest Correlation Matrix''   #################
#################                           By Houduo Qi and Defeng Sun                                        #################
#################                     SIAM J. Matrix Anal. Appl. 28 (2006) 360--385.                           #################
#################                                                                                              #################
################################################################################################################################
#################                                                                                              #################
#################   The input arguments  G, b>0, tau>=0, and tol (tolerance error)                             #################
#################                                                                                              #################
#################        For the correlation matrix problem, set b = rep(1,n)                                  #################
#################                                                                                              #################
#################        For a positive definite matrix                                                        #################
#################            set tau = 1.0e-5 for example                                                      #################
#################            set tol = 1.0e-6 or lower if no very high accuracy required                       #################
#################                      #############################################                           #################
#################   If the algorithm terminates with the given tol, then it means                              #################
#################         the relative gap between the  objective function value of the obtained solution      #################
#################         and the objective function value of the global solution is no more than tol or       #################
#################         the violation of the feasibility is smaller than tol*(1+norm(b,'2'))                 #################
#################                      #############################################                           #################
#################   The outputs include the optimal primal solution, the dual solution and others              #################
#################                           Diagonal Preconditioner is used                                    #################
#################                                                                                              #################
################################################################################################################################
#################      Send your comments to hdqi@soton.ac.uk  or defeng.sun@polyu.edu.hk                      #################
#################                                                                                              #################
#################                              Last modified date: August 31, 2019 by Qian LI
#################                                                     at 18090081g@connect.polyu.hk            ###############                                              Note:  For faster calculation, you may use MRO. #################
#################     Warning:  Though the code works extremely well, it is your call to use it or not.        #################

# The code is obtained from https://www.polyu.edu.hk/ama/profile/dfsun/
# Authors and references are desribed above.
# Modified by Xiangjie Xue.
# Remove the code for printing and timing.

CorrelationMatrix <- function(G,b,tau=0,tol=1e-6){
  n <- dim(G)[1]

  if (missing(b) )
  {
    b = rep(1,n)
  }
  if (tau>0)
  {
    b= b - tau
    G  =  G - tau*diag(n)
  }

  b0 <- as.matrix(b)
  error_tol <- max(1e-12,tol)
  Res_b <- array(0,c(300,1))
  y <- array(0,c(n,1))         #### initial dual value
  Fy <- array(0,c(n,1))

  Iter_whole <- 200
  Iter_inner <- 20
  Iter_CG <- 200

  f_eval <- 0
  iter_k <- 0
  iter_linesearch <- 0
  tol_CG <- 1e-2               #### relative accuracy of the conjugate gradient method for solving the Newton direction
  G_1 <- 1e-4                  #### tolerance in the line search of the Newton method

  x0 <- y
  c <- array(1,c(n,1))
  d <- array(0,c(n,1))
  val_G <- 0.5*norm(G,'F')^2
  time_eig <- 0
  time_pcg <- 0
  time_prec <- 0

  if (n == 1)
  {
    X <- G + y
  }
  else
  {
    X <- G + diag(c(y))
  }
  X <- (X + t(X))/2
  Eigen_X <- Myeigen(X,n)

  P <- Eigen_X$vectors
  lambda <- Eigen_X$values
  gradient <- Corrsub_gradient(y,lambda,P,b0,n)
  f0 <- gradient$f
  Fy <- gradient$Fy
  val_dual <- val_G - f0
  X <- PCA(X,lambda,P,b0,n)
  val_obj <- 0.5*norm(X-G,'F')^2
  gap <- (val_obj - val_dual)/(1+abs(val_dual) + abs(val_obj))

  f <- f0
  f_eval <- f_eval + 1
  b <- b0 - Fy
  norm_b <- norm(b,'2')
  norm_b0 <- norm(b0,'2')+1
  norm_b_rel <- norm_b/norm_b0
  Omega12 <- omega_mat(lambda,n)

  x0 <- y
  while (gap > error_tol & norm_b_rel > error_tol & iter_k< Iter_whole) {
    c <- precond_matrix(Omega12,P,n)

    CG_result<- pre_cg(b,tol_CG,Iter_CG,c,Omega12,P,n)

    d <- CG_result$dir

    slope <- sum((Fy - b0)*d)

    y <- x0+d
    if (n == 1){
      X <- G + y
    }
    else{
      X <- G + diag(c(y))
    }

    X <- (X+t(X))/2
    Eigen_X <- Myeigen(X,n)
    P <- Eigen_X$vectors
    lambda <- Eigen_X$values

    gradient <- Corrsub_gradient(y,lambda,P,b0,n)
    f <- gradient$f
    Fy <- gradient$Fy
    k_inner <- 0

    while(k_inner <= Iter_inner & f>f0 + G_1*0.5^k_inner*slope +1e-6)
    {
      k_inner <- k_inner+1
      y <- x0 + 0.5^k_inner*d
      X <- G + diag(c(y))
      X <- (X + t(X))/2
      Eigen_X <- Myeigen(X,n)
      P <- Eigen_X$vectors
      lambda <- Eigen_X$values
      gradient <- Corrsub_gradient(y,lambda,P,b0,n)
      f <- gradient$f
      Fy <- gradient$Fy
    }

    f_eval <- f_eval + k_inner + 1
    x0 <- y
    f0 <- f
    val_dual <- val_G  - f0
    X <- PCA(X,lambda,P,b0,n)
    val_obj <- 0.5*norm(X-G,'F')^2
    gap <- (val_obj - val_dual)/(1+abs(val_dual)+abs(val_obj))

    iter_k <- iter_k + 1
    b <- b0 - Fy
    norm_b <- norm(b,'2')
    Res_b[iter_k] <-  norm_b
    norm_b_rel <- norm_b/norm_b0

    Omega12 <- omega_mat(lambda,n)
  }

  X <- X + tau*diag(n)
  Final_f <- val_G - f
  rank_X <- sum(lambda >= 0)

  info <- list(CorrMat = X, dual_sol = y,rel_grad = norm_b_rel, rank = rank_X, gap = gap,iterations = iter_k)

  return(info)

}



########################################################################################

#Generate the eigenvalures and corresponding eigenvectors of a symmetrix matrix X
#noted that the output eigenvalues is a vector sorted in deccreasing order automaticly by using eigen function

Myeigen <- function(X,n){

  Eigen_X <- eigen(X,symmetric = TRUE)
  P <- Eigen_X$vectors
  lambda <- Eigen_X$values
  if (sum(lambda >0)==0){
    cat('\n Warning: no positive eigenvalue\n')
  }
  list(values = lambda, vectors = P)
}

#Generate F(y):=diag(G+diag(y))+
#And Generate part of the dual value denoted by f

Corrsub_gradient <- function(y,lambda,P,b,n){
  r <- sum(lambda >0)
  if (r>0){
    P1 <- P[,1:r]
    if (r ==1){
      P1 <- matrix(P1,n,1)
    }
    lam1 <- lambda[1:r]
    Fy <- matrix(0,n,1)+rowSums(as.matrix(sweep(P1,2,lam1,'*'))*P1)
    f=0.5*sum(lam1*lam1)-sum(b*y)
  }else{
    Fy <- matrix(0,n,1)
    f <- 0
  }

  list(f = f, Fy = Fy)
}

#Use PCA to generate a primal feasible solution


PCA <- function(X,lambda,P,b,n){
  if (n==1){
    X <- b
  }
  else{
    r <- sum(lambda >0)
    if (r>1 && r<n){
      if (r <= 2){
        P1 <- P[,1:r]
        lam1 <- sqrt(lambda[1:r])
        P1lam1s <- sweep(P1,2,lam1,'*')
        X <- tcrossprod(P1lam1s)
      }
      else{
        P2 <- as.matrix(P[,(r+1):n])
        lam2 <- sqrt(-lambda[(r+1):n])
        P2lam2s <- sweep(P2,2,lam2,'*')
        X=X+tcrossprod(P2lam2s)
      }
    }
    else{
      if (r == 0){
        X <- mat.or.vec(n, n)
      }
      else if (r == n) {                  #find A feasible solution X
        X <- X                          #that is positive semi-definite firstly
      }
      else if (r == 1){
        X <- lam1[1]^2*tcrossprod(P1)
      }
    }
    d <- diag(X)                   #make the diagonal vector to be b without changing psd propertity
    d <- pmax(d,b)
    X <- X - diag(diag(X)) + diag(d)
    d <- (b/d)^0.5
    d2 <- tcrossprod(d)
    X <- X*d2
  }
  return(X)
}

#Generate the second block of M(y)
#the essential part of the first -order difference of d

omega_mat <- function(lambda,n){
  r <- sum(lambda >0)

  if (r>0)
  {
    if (r<n)
    {
      s <- n-r
      Omega12 <- matrix(1,r,s)
      Omega12 <- apply(matrix(1,nrow = r,ncol = s),2,function(x){x*lambda[1:r]})
      Omega12 <- sweep(Omega12,2,lambda[(r+1):n],function(x,y){x/(x-y)})
    }
    else
    {
      Omega12 <- matrix(1,n,n)
    }
  }
  else
  {
    Omega12 <- matrix(nrow=0, ncol=0)
  }
  return(Omega12)
}


#To generate the Jacobain product with d: F'(y)(d)=V(y)d

Jacobian_matrix <- function(d,Omega12,P,n){
  r <- dim(Omega12)[1]
  Vd <- array(0,c(n,1))
  if (r>0){
    if (r < n){
      P1 <- P[,1:r]
      P2 <- as.matrix(P[,(r+1):n])
      O12 <- Omega12*(t(P1)%*%sweep(P2,1,d,'*')) #O12=Omega12.*(P1'D P2)  where D=diag(d)
      PO <- P1%*%O12
      hh <- 2*rowSums(PO*P2)  #hh=diag(PO%*%P2)
      if (r <= n/2){
        PP1 <- tcrossprod(P1)
        Vd <- apply(PP1,1,function(x){sum(x^2*d)})+hh+1e-10*d
      }
      else{
        PP2 <- tcrossprod(P2)
        Vd <- d+apply(PP2,1,function(x){sum((x^2)*d)})+hh-(2*d*diag(PP2))+1e-10*d
      }
    }
    else{
      Vd <- (1+1e-10)*d
    }
  }
  return(Vd)
}

#Generate the diginal Precondition
precond_matrix <- function(Omega12,P,n){
  r <- dim(Omega12)[1]
  c <- array(1,c(n,1))
  if (r>1){
    H <- t(P)
    H <- H*H
    H1 <- H[1:r,]
    if (r<n/2){
      H2 <- as.matrix(H[(r+1):n,])
      H12 <- crossprod(H1,Omega12)
      c <- colSums(H1)^2+2*rowSums(H12*t(H2))
    }
    else{
      if (r<n){
        H2 <- as.matrix(H[(r+1):n,])
        H12 <- (1-Omega12)%*%H[(r+1):n,]
        c <- colSums(H)^2-colSums(H2)^2-2*rowSums(t(H1*H12))
      }

    }
  }
  else if (r == 1){
    H1 <- matrix(H1, 1,n)
  }
  c=max(c,1e-8)
  return(c)
}

#PCG method :An iterative method to solve A(x) =b
#this is proposed by Hestenes and Stiefel (1952)

pre_cg <- function(b,tol,Iter_CG,c,Omega12,P,n){
  r <- b  ### initial value for CG: zero
  n2b <- norm(b,'2')
  tolb <- tol*n2b
  p <- array(0,c(n,1))
  flag <- 1
  iterk <- 0
  relres <- 1000
  z <- r/c  ####z = M\r; here M =diag(c); if M is not the identity matrix
  rz1 <- sum(r*z)
  rz2 <- 1
  d <- z
  for (k in 1:Iter_CG) {
    if (k > 1)
    {
      beta <- rz1/rz2
      d <- z + beta*d
    }
    w <- Jacobian_matrix(d,Omega12,P,n)  #w=V(y)*d
    denom <- sum(d*w)
    iterk <- k
    relres <- norm(r,'2')/n2b
    if (denom<= 0){
      p <- d/norm(d)
      break
    }
    else{
      alpha <- rz1/denom
      p <- p+alpha*d
      r <- r- alpha*w
    }
    z <- r/c
    if(norm(r,'2')<= tolb){
      iterk <- k
      relres <- norm(r,'2')/n2b
      flag <- 0
      break
    }
    rz2 <- rz1
    rz1 <- sum(r*z)
  }
  list(dir=p,flag=flag,iter=iterk)
}
