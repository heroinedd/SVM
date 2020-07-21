package svm;

import java.util.ArrayList;
import java.util.Random;

public class SVM_SMO {
	public int num_train_sample;
	public int x_dim;
	public int num_test_sample;
	public double epsilon;//精度
	public double tol;
	public double C;//0<=alpha[i]<=C
	
	
	public double [][] train_sample_x;
	public int [] train_sample_y;
	public double [][] test_sample_x;
	public int [] test_sample_y;
	//训练数据和测试数据
	
	
	public double [] alpha;
	public double [] w;
	public double b_old,b_new;//SMO原论文中是按wx-b来的，下面所有的公式都是按照-b推导的
	public double [] fx;
	public ArrayList<Integer> support_vector;
	public ArrayList<Integer> nonbound_alpha;
	
	public double [][] K;
	public double alpha1_old,alpha2_old;
	public double alpha1_new,alpha2_new;
	public double [] E;//E[i]=u[i]-y[i] is the error on the ith training example
	public double L;
	public double H;
	public double alpha2_temp;
	
	public static double bandwidth;
	public static double c=-2*bandwidth*bandwidth;
	
	SVM_SMO(){
		num_train_sample=60;
		num_test_sample=100;
		x_dim=28*28;
		epsilon=0.001;
		tol=0.001;
		C=100;
		train_sample_x=new double [num_train_sample][x_dim];
		train_sample_y=new int[num_train_sample];
		test_sample_x=new double[num_test_sample][x_dim];
		test_sample_y=new int[num_test_sample];
		alpha=new double[num_train_sample];
		w=new double[x_dim];
		fx=new double[num_train_sample];
		support_vector=new ArrayList<Integer>(num_train_sample);
		nonbound_alpha=new ArrayList<Integer>(num_train_sample);
		K=new double[num_train_sample][num_train_sample];
		E=new double[num_train_sample];
	}
	SVM_SMO(int train,int test,int dim,double eps,double to,double c){
		num_train_sample=train;
		num_test_sample=test;
		x_dim=dim;
		epsilon=eps;
		tol=to;
		C=c;
		train_sample_x=new double [num_train_sample][x_dim];
		train_sample_y=new int[num_train_sample];
		test_sample_x=new double[num_test_sample][x_dim];
		test_sample_y=new int[num_test_sample];
		alpha=new double[num_train_sample];
		w=new double[x_dim];
		fx=new double[num_train_sample];
		support_vector=new ArrayList<Integer>(num_train_sample);
		nonbound_alpha=new ArrayList<Integer>(num_train_sample);
		K=new double[num_train_sample][num_train_sample];
		E=new double[num_train_sample];
	}
	
	public void initial() {
		b_old=0;
		b_new=0;
		/*
		 * 初始置所有的alpha=0
		 */
		for(int i=0;i<num_train_sample;i++) {
			alpha[i]=0;
			w[i]=0;
			fx[i]=0;
			E[i]=-train_sample_y[i];
		}
		for(int i=0;i<num_train_sample;i++) {
			K[i][i]=linear_kernel(train_sample_x[i],train_sample_x[i]);
			for(int j=i+1;j<num_train_sample;j++) {
				K[i][j]=linear_kernel(train_sample_x[i],train_sample_x[j]);
				K[j][i]=K[i][j];
			}
		}
		System.out.println("initial done");
	}
	
	public void update_w(int i,int j) {
		for(int m=0;m<x_dim;m++) {
			w[m]=w[m]+train_sample_y[i]*train_sample_x[i][m]*(alpha1_new-alpha1_old)
					+train_sample_y[j]*train_sample_x[j][m]*(alpha2_new-alpha2_old);
		}
	}
	
	public void update_E_fx(int i,int j) {
		for(int m=0;m<num_train_sample;m++) {
			fx[m]=fx[m]+train_sample_y[i]*(alpha1_new-alpha1_old)*K[m][i]+train_sample_y[j]*(alpha2_new-alpha2_old)*K[m][j]+(b_old-b_new);
			E[m]=fx[m]-train_sample_y[m];
		}
	}
	
	public int takestep(int i,int j) {
		if(i==j) return 0;
		alpha1_old=alpha[i];
		alpha2_old=alpha[j];
		double eta=K[i][i]+K[j][j]-2*K[i][j];
		int s=train_sample_y[i]*train_sample_y[j];
		double f1,f2,L1,H1,Psi_L,Psi_H;
		
		if(train_sample_y[i]!=train_sample_y[j]) {
			L=Math.max(0, alpha2_old-alpha1_old);
			H=Math.min(C, C+alpha2_old-alpha1_old);
		}
		else {
			L=Math.max(0, alpha2_old+alpha1_old-C);
			H=Math.min(C, alpha2_old+alpha1_old);
		}
		if(L==H) return 0;
		
		if(eta>0) {
			alpha2_temp=alpha2_old+train_sample_y[j]*(E[i]-E[j])/eta;
			if(alpha2_temp<=L) alpha2_new=L;
			if(L<alpha2_temp&&alpha2_temp<H) alpha2_new=alpha2_temp;
			if(H<=alpha2_temp) alpha2_new=H;
		}
		
		else {
			f1=train_sample_y[i]*(E[i]+b_new)-alpha1_old*K[i][i]-s*alpha2_old*K[i][j];
			f2=train_sample_y[j]*(E[j]+b_new)-s*alpha2_old*K[i][j]-alpha2_old*K[j][j];
			L1=alpha1_old+s*(alpha2_old-L);
			H1=alpha1_old+s*(alpha2_old-H);
			Psi_L=L1*f1+L*f2+1/2*L1*L1*K[i][i]+1/2*L*L*K[j][j]+s*L*L1*K[i][j];
			Psi_H=H1*f1+H*f2+1/2*H1*H1*K[i][i]+1/2*H*H*K[j][j]+s*H*H1*K[i][j];
			if(Psi_L<Psi_H-epsilon) alpha2_new=L;
			else if(Psi_L>Psi_H+epsilon) alpha2_new=H;
			else alpha2_new=alpha2_old;
		}
		
		if(Math.abs(alpha2_temp-alpha2_old)<epsilon*(alpha2_temp+alpha2_old+epsilon)) return 0;
		alpha1_new=alpha1_old+s*(alpha2_old-alpha2_new);
		/*
		 * update threshold、weight vector、error cache
		 */
		update_b(i,j);
		update_w(i,j);
		update_E_fx(i,j);
		
		alpha[i]=alpha1_new;
		alpha[j]=alpha2_new;
		if(alpha[i]!=0) {
			if(!support_vector.contains(i)) support_vector.add(i);
			if(!nonbound_alpha.contains(i)&&alpha[i]!=C) nonbound_alpha.add(i);
		}
		if(alpha[j]!=0) {
			if(!support_vector.contains(j)) support_vector.add(j);
			if(!nonbound_alpha.contains(j)&&alpha[j]!=C) nonbound_alpha.add(j);
		}
		//System.out.println(support_vector.size()+"\t"+nonbound_alpha.size());
		return 1;
	}
	
	public void update_b(int i,int j) {
		b_old=b_new;
		double b1_new,b2_new;
		b1_new=E[i]+(alpha1_new-alpha1_old)*train_sample_y[i]*K[i][i]+(alpha2_new-alpha2_old)*train_sample_y[j]*K[i][j]+b_new;
		b2_new=E[j]+(alpha1_new-alpha1_old)*train_sample_y[i]*K[j][i]+(alpha2_new-alpha2_old)*train_sample_y[j]*K[j][j]+b_new;
		boolean in1,in2;
		in1=0<alpha1_new&&alpha1_new<C?true:false;
		in2=0<alpha2_new&&alpha2_new<C?true:false;
		if(in1||in2) b_new=in1?b1_new:b2_new;
		else b_new=(b1_new+b2_new)/2;
	}
	
	/*
	 * 内层循环
	 * under usual circumstances, SMO makes positive progress 
	 * under unusual circumstances, SMO cannot make positive progress
	 * in this case, SMO uses a hierarchy of second choice heuristics until it 
	 * finds a pair of Lagrange multipliers that can make positive progress
	 */
	public int examineExample(int j) {
		int y2=train_sample_y[j];
		double alpha2=alpha[j];
		double E2=E[j];
		int sign=E2>0?1:-1;
		double r2=E2*y2;
		double E1;
		if((r2<-tol&&alpha2<C)||(r2>tol&&alpha2>0)) {
			//System.out.println("a");
			if(nonbound_alpha.size()>1) {
				int i=nonbound_alpha.get(0);
				int index=i;
				E1=E[index];
				double temp=sign*E1;
				for(int m=1;m<nonbound_alpha.size();m++) {
					index=nonbound_alpha.get(m);
					E1=E[index];
					if(sign*E1<temp) {
						temp=sign*E1;
						i=index;
					}
				}
				if(takestep(i,j)==1) {
					//System.out.println("a");
					return 1;
				}
			}
			Random rand=new Random();
			int begin;
			if(nonbound_alpha.size()>1) {
				int size=nonbound_alpha.size();
				begin=rand.nextInt(size);
				for(int n=0;n<size;n++) {
					int i=nonbound_alpha.get((n+begin)%size);
					if(takestep(i,j)==1) {
						//System.out.println("b");
						return 1;
					}
				}
			}
			begin=rand.nextInt(num_train_sample);
			for(int n=0;n<num_train_sample;n++) {
				int i=(n+begin)%num_train_sample;
				if(nonbound_alpha.contains(i)) continue;
				if(takestep(i,j)==1) {
					//System.out.println("c");
					return 1;
				}
			}
		}
		//System.out.println("d");
		return 0;
	}
	
	/*
	 * SMOmain函数实际上是外层循环
	 * 摘自原论文
	 * The outer loop keeps alternating between single passes over the entire training set
	 * and multiple passes over the non-bound subset until the entire training set obeys
	 * the KKT conditions within epsilon, where upon the algorithm terminates
	 * 
	 * examineAll: single passes over the entire training set
	 * numChanged>0: multiple passes over the non-bound subset
	 */
	public void SMOmain() {
		int numChanged=0;
		boolean examineAll=true;
		while(numChanged>0||examineAll) {
			//System.out.println("outer loop");
			numChanged=0;
			if(examineAll) {
				for(int m=0;m<num_train_sample;m++)
					numChanged+=examineExample(m);
			}
			else {
				for(int m=0;m<nonbound_alpha.size();m++)
					numChanged+=examineExample(nonbound_alpha.get(m));
			}
			//System.out.println(numChanged);
			if(examineAll) examineAll=false;
			else if(numChanged==0) examineAll=true;
		}
	}
	
	public double correct_rate() {
		double rate=0;
		for(int i=0;i<num_train_sample;i++) {
			System.out.print(train_sample_y[i]+" "+fx[i]+"\n");
			if(fx[i]*train_sample_y[i]>0) rate++;
		}
		return rate/num_train_sample;
	}
	
	//各种核函数
	public static double linear_kernel(double[] x1,double[] x2) {
		double sum=0.0;
		for(int i=0;i<x1.length;i++) {
			sum+=x1[i]*x2[i];
		}
		return sum;
	}
	
	public static double polynomial_kernel(double d,double[] x1,double[] x2) {
		double sum=0.0;
		for(int i=0;i<x1.length;i++) {
			sum+=x1[i]*x2[i];
		}
		sum=Math.pow(sum, d);
		return sum;
	}
	
	public static double gaussian_kernel(double[] x1,double[] x2) {
		double sum=0.0;
		for(int i=0;i<x1.length;i++) {
			sum+=(x1[i]-x2[i])*(x1[i]-x2[i]);
		}
		sum/=c;
		sum=Math.exp(sum);
		return sum;
	}
	
	public static double laplace_kernel(double[] x1,double[] x2) {
		double sum=0.0;
		for(int i=0;i<x1.length;i++) {
			sum+=(x1[i]-x2[i])*(x1[i]-x2[i]);
		}
		sum=Math.pow(sum, 1/2);
		sum/=-bandwidth;
		sum=Math.exp(sum);
		return sum;
	}
	
//	public static double sigmoid_kernel() {
//		
//	}

}
