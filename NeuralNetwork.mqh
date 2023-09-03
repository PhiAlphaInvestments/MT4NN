
#property strict 


//inspired by,
//https://github.com/stephencwelch/Neural-Networks-Demystified




class CRowDouble;

int ArrayResizeAL(CRowDouble &arr[],const int size)
  {

   return(ArrayResize(arr,size));
  }
int ArrayResizeAL(double &arr[],const int size)
  {

   return(ArrayResize(arr,size));
  }


class CRowDouble
  {
private:
   double            m_array[];

public:
                     CRowDouble(void);
                    ~CRowDouble(void);
   
   int               Size(void) const;
   void              Resize(const int n);
   void              Set(const int i,const double d);
   
   double            operator[](const int i) const;
   void              operator=(const double &array[]);
   void              operator=(const CRowDouble &r);
  };

CRowDouble::CRowDouble(void)
  {

  }

CRowDouble::~CRowDouble(void)
  {

  }

int CRowDouble::Size(void) const
  {
   return(ArraySize(m_array));
  }

void CRowDouble::Resize(const int n)
  {
   ArrayResizeAL(m_array,n);
  }

void CRowDouble::Set(const int i,const double d)
  {
   m_array[i]=d;
  }

double CRowDouble::operator[](const int i) const
  {
   return(m_array[i]);
  }

void CRowDouble::operator=(const double &array[])
  {
   int size=ArraySize(array);

   if(size==0)
      return;

   ArrayResizeAL(m_array,size);
   for(int i=0;i<size;i++)
      m_array[i]=array[i];
  }

void CRowDouble::operator=(const CRowDouble &r)
  {
   int size=r.Size();

   if(size==0)
      return;

   ArrayResizeAL(m_array,size);
   for(int i=0;i<size;i++)
      m_array[i]=r[i];
  }

class CMatrixDouble
  {
private:
   
   CRowDouble        m_rows[];
   int               Col;
   int               Row;
   CRowDouble        temp[];
public:
   
                     CMatrixDouble(void);
                     CMatrixDouble(const int rows);
                     CMatrixDouble(const int rows,const int cols);
                     CMatrixDouble(const CMatrixDouble &m); 
                    ~CMatrixDouble(void);
   
   void              Insert(int Row, int Col, double val);
   double            Sum(void);
   double            Mean(void);
   CMatrixDouble     Fill(double val);
   int               SizeRow(void) const;
   int               SizeCol(void) const;
   void              Resize(const int n,const int m);
   CMatrixDouble     Transpose(void);
   void              PrintMatrix(void);
   
   
   CRowDouble       *operator[](const int i) const;
   void              operator=(const CMatrixDouble &m);
   CMatrixDouble             operator+(const CMatrixDouble &m);
   CMatrixDouble              operator*(const CMatrixDouble &m);
   CMatrixDouble              operator-(const CMatrixDouble &m);
  };

void CMatrixDouble::Insert(int Row_,int Col_,double val_)
 {
  
  
  
  const double val_c = val_;
   
  
      
        m_rows[Row_].Set(Col_, val_c);
      
  
  
  
  

  
  }

double CMatrixDouble::Sum(void)
 {
  
  double sum =0;
  
  
   
  
   for( int i = 0 ; i<Row ; i++){
      
      for( int j = 0 ; j <Col ; j++){
      
      
        sum= sum + m_rows[i][j];
      
      }
      
   
   }
  
  
  

  return sum; 
  }



double CMatrixDouble::Mean(void)
 {
  
  double sum =0;
  
  
   
  
   for( int i = 0 ; i<Row ; i++){
      
      for( int j = 0 ; j <Col ; j++){
      
      
        sum= sum + m_rows[i][j];
      
      }
      
   
   }
  
  
  

  return sum/(Row*Col); 
  }


CMatrixDouble CMatrixDouble::Fill(double val)
 {
  
  CMatrixDouble A(Row,Col);
  
  
   for( int i = 0 ; i<Row ; i++){
      
      for( int j = 0 ; j <Col ; j++){
      
      
         A[i].Set(j,  val );
      
      }
      
   
   }
  
  
  
 
  return A; 
  }  

CMatrixDouble::CMatrixDouble(void)
  {
  Row = 0;
  Col = 0;

  }

CMatrixDouble::CMatrixDouble(const int rows)
  {
   ArrayResizeAL(m_rows,rows);
   Row = rows;
   Col = 0; 
  }

CMatrixDouble::CMatrixDouble(const int rows,const int cols)
  {
  Row = rows;
  Col = cols; 
   ArrayResizeAL(m_rows,rows);
   for(int i=0;i<rows;i++)
      m_rows[i].Resize(cols);
  }

CMatrixDouble::~CMatrixDouble(void)
  {

  }

int CMatrixDouble::SizeRow(void) const
  {
   return(Row);
  }
  
  
int CMatrixDouble::SizeCol(void) const
  {
   return(Col);
  }  

void CMatrixDouble::Resize(const int n,const int m)
  {

   if(n<0 || m<0)
      return;

   ArrayResizeAL(m_rows,n);
   for(int i=0;i<n;i++)
      m_rows[i].Resize(m);
   Row = n;
   Col = m;
  }

CRowDouble *CMatrixDouble::operator[](const int i) const
  {
   return(GetPointer(m_rows[i]));
  }

CMatrixDouble CMatrixDouble::operator+(const CMatrixDouble &m)
  {
  
  CMatrixDouble A(Row,Col);
  if( m.SizeRow() == Row && m.SizeCol() == Col ){
  
   
  
   for( int i = 0 ; i<Row ; i++){
      
      for( int j = 0 ; j <Col ; j++){
      
      
         A[i].Set(j, m[i][j] + m_rows[i][j]);
      
      }
      
   
   }
  
  
  
  }
  else{
  
   Print("Failed Addition");  
  
  
  }
  return A; 
  }

CMatrixDouble CMatrixDouble::operator-(const CMatrixDouble &m)
  {
  
  CMatrixDouble A(Row,Col);
  if( m.SizeRow() == Row && m.SizeCol() == Col ){
  
   
  
   for( int i = 0 ; i<Row ; i++){
      
      for( int j = 0 ; j <Col ; j++){
      
      
         A[i].Set(j,  m_rows[i][j]-m[i][j] );
      
      }
      
   
   }
  
  
  
  }
  else{
  
   Print("Failed Subtraction");  
  
  
  }
  return A; 
  }  
  
  

CMatrixDouble CMatrixDouble::operator*(const CMatrixDouble &m)
  {
  CMatrixDouble A(Row,Col);
  if( m.SizeRow() == Row && m.SizeCol() == Col ){
  
  
   for( int i = 0 ; i<Row ; i++){
      
      for( int j = 0 ; j <Col ; j++){
      
      
         A[i].Set(j, m[i][j] * m_rows[i][j]);
      
      }
      
   
   }
  
  
  
  }
  else{
  
   Print("Failed Multiplication");  
  
  }
   return A;
  }



void CMatrixDouble::operator=(const CMatrixDouble &m)
  {
   Row = m.SizeRow();
   Col = m.SizeCol();
   int r=m.SizeRow();

   if(r==0)
      return;
   int c=m[0].Size();

   if(c==0)
      return;

   ArrayResizeAL(m_rows,r);
   for(int i=0;i<r;i++)
      m_rows[i].Resize(c);

   for(int i=0;i<r;i++)
      m_rows[i]=m[i];
  }

void CMatrixDouble::CMatrixDouble(const CMatrixDouble &m){
   Row = m.SizeRow();
   Col = m.SizeCol();

   int r=m.SizeRow();

   if(r==0)
      return;
   int c=m[0].Size();

   if(c==0)
      return;

   ArrayResizeAL(m_rows,r);
   for(int i=0;i<r;i++)
      m_rows[i].Resize(c);

   for(int i=0;i<r;i++)
      m_rows[i]=m[i];
  }
CMatrixDouble CMatrixDouble::Transpose(void){

      CMatrixDouble temp1;

      if(Row != 0 || Col != 0){
      
      int RowTemp = Row;
      int ColTemp = Col;
      
      
     temp1.Resize(Col,Row);
     
      
      for (int i = 0; i < Col; ++i){
         for (int j = 0; j < Row; ++j) {
               
               
             temp1[i].Set( j ,m_rows[j][i]);
            
         }
       }
      
      
      }
      else{
      
      //pass
      
      }
      
      return temp1;

}



void CMatrixDouble::PrintMatrix(void){
   
   
      
      string out ;
      
      out += "(";
      out += IntegerToString(Row);
      out += ",";
      out += IntegerToString(Col);
      out += ")"; 
      if(Row != 0 || Col != 0){
      
      
      out += "[";
      for (int i = 0; i < Row; ++i){
         out += "[";
         for (int j = 0; j < Col; ++j) {
               
               
            out +=  DoubleToStr( m_rows[i][j]) + ",";
            
         }
         out += "]";
       }
      
      out += "]";
      
      Print( out);
      }
      else{
      
      //pass
      
      }


}



//https://www.tutorialspoint.com/generate-random-numbers-following-a-normal-distribution-in-c-cplusplus
double rand_gen() {
   // return a uniformly distributed random value
   return ( (double)(rand()) + 1. )/( (double)(32767) + 1. );
}
double normalRandom() {
   // return a normally distributed random value
   double v1=rand_gen();
   double v2=rand_gen();
   return cos(2*3.14*v2)*sqrt(-2.*log(v1));
}
double NormalDistSample( double Mu, double sigma){
  
   
      double x = normalRandom()*sigma+Mu;
      return x;    
}



 
class NeuralNetwork{

   protected:
         int m_maxiters;
         double m_beta_1;
         double m_beta_2;
         bool m_verbose;
         double m_LearningRate;
         int m_deep;
         int m_depth;
         CMatrixDouble m_input;
         CMatrixDouble m_pred_input;
         CMatrixDouble m_z_2;
         CMatrixDouble m_a_2; 
         CMatrixDouble m_z_3;
         CMatrixDouble m_yHat;
         CMatrixDouble z_3_prime;
         CMatrixDouble z_2_prime;
         CMatrixDouble delta2;
         CMatrixDouble delta3;
         CMatrixDouble dJdW1;
         CMatrixDouble dJdW2;
         CMatrixDouble y_cor;
         double m_alpha;
         int m_outDim;
         
         
         CMatrixDouble Forward_Prop(CMatrixDouble &Input);
         double Cost(CMatrixDouble &Input, CMatrixDouble &y_cor);
         double Sigmoid(double x);
         double Sigmoid_Prime(double x);     
         void   MatrixRandom(CMatrixDouble &m);
         CMatrixDouble MatrixSigmoidPrime(CMatrixDouble &m);
         CMatrixDouble MatrixSigmoid(CMatrixDouble &m);
         void   ComputeDerivatives(CMatrixDouble &Input , CMatrixDouble &y_);
         CMatrixDouble MatrixMultiply( CMatrixDouble &m1, CMatrixDouble &m2);
   
      public:
   
      
         CMatrixDouble W_1;
         CMatrixDouble W_2;
         
         NeuralNetwork(int in_DimensionRow,int in_DimensionCol,int Number_of_Neurons,int out_Dimension,double alpha,double LearningRate,bool Verbose,double beta_1, double beta_2,int max_iterations);
         void   Train(CMatrixDouble &Input,CMatrixDouble &correct_Val); 
         int    Sgn(double Value);
         CMatrixDouble Prediction(CMatrixDouble &Input); 
         void   ResetWeights();
         bool   WriteWeights();
         bool   LoadWeights();

};


CMatrixDouble NeuralNetwork::MatrixMultiply( CMatrixDouble& m1, CMatrixDouble& m2){

    
    CMatrixDouble A(m1.SizeRow(),m2.SizeCol());
    
    
    if( m1.SizeCol() == m2.SizeRow()){
    
    
    
    int N = m1.SizeRow();
    int C = m2.SizeCol();
    int B  = m1.SizeCol();
    for( int i = 0; i < N ; i++){
    
      for( int j =0 ; j < C ; j++){
      
         double product = 0;
         for( int k = 0; k< B ; k++){
         
            product += m1[i][k]*m2[k][j];               
         
         } 
         
         A[i].Set(j,product);
      
      } 
      
    
    
    
    
    } 
  
   
    
    
    
    }
    else{
    
    Print( "Failed Matrix Multiply");
    m1.PrintMatrix();
    m2.PrintMatrix();
    }  
    /* 
    
  
      */
        
      
         
   
      return A;







}




bool NeuralNetwork::LoadWeights(void){

      
      
         
       //int handle = FileOpen("Weights_1.txt",FILE_READ,",",FILE_TXT);
       //to be dev
         
         
         return true;

}
bool NeuralNetwork::WriteWeights(void){
      /*
      string InpName = "Weights_1.txt";

      int handle_w1=FileOpen(InpName,FILE_READ|FILE_WRITE|FILE_CSV);
      
      

      
      InpName = "Weights_2.txt";

      int handle_w2=FileOpen(InpName,FILE_READ|FILE_WRITE|FILE_TXT);
      
      
      
      
      FileWrite(handle_w2,W_2 );
      FileClose(handle_w2);
      */
      //to be dev
      
      return true;
};

void NeuralNetwork::ResetWeights(void){

     
       
       CMatrixDouble random_W1(m_depth, m_deep);
       CMatrixDouble random_W2(m_deep, m_outDim);
       
       MatrixRandom(random_W1);
       MatrixRandom(random_W2);
       
       W_1      =   random_W1;
       W_2      =   random_W2; 


}


void NeuralNetwork::ComputeDerivatives(CMatrixDouble &Input , CMatrixDouble &y_){

       CMatrixDouble X = Input;
       CMatrixDouble Y = y_;  
         
        m_yHat = Forward_Prop(X); 
        
        //Print( m_yHat.Cols(),m_yHat.Rows() );
         
         
        CMatrixDouble neg1 = m_yHat; 
        
        neg1=neg1.Fill(-1);
        
        CMatrixDouble cost =(Y-m_yHat);
        cost = neg1*cost;
        
        z_3_prime = MatrixSigmoidPrime(m_z_3);
        
        delta3 = cost*(z_3_prime);
       
       
        CMatrixDouble m_a_2_T = m_a_2.Transpose();
        dJdW2 =   MatrixMultiply(m_a_2_T,delta3); 
        
        
        z_2_prime = MatrixSigmoidPrime(m_z_2);
        CMatrixDouble W_2_T =W_2.Transpose();
        
        delta2 =  MatrixMultiply(delta3,W_2_T);          
        delta2 = delta2*z_2_prime;
        
        CMatrixDouble m_input_T = m_input.Transpose();
        
        dJdW1 = MatrixMultiply(m_input_T,delta2);
        
        


};


NeuralNetwork::NeuralNetwork(int in_DimensionRow,int in_DimensionCol,int Number_of_Neurons,int out_Dimension,double alpha,double LearningRate,bool Verbose, double beta_1, double beta_2,int max_iterations) {
       
       m_depth = in_DimensionCol;
       m_deep  = Number_of_Neurons;
       m_alpha = alpha;
       m_outDim= out_Dimension;
       m_LearningRate = LearningRate;
       m_beta_1 = beta_1;
       m_beta_2 = beta_2;
       CMatrixDouble random_W1(m_depth, m_deep);
       CMatrixDouble random_W2(m_deep, out_Dimension);
       
       m_verbose = Verbose;
       m_maxiters =max_iterations;
       MatrixRandom(random_W1);
       MatrixRandom(random_W2);
       
       W_1      =   random_W1;
       W_2      =   random_W2; 
       W_1.PrintMatrix();
       W_2.PrintMatrix();
       
       
       
       
       }




CMatrixDouble NeuralNetwork::Prediction(CMatrixDouble& Input){
   
   m_pred_input = Input;
       
   CMatrixDouble pred_z_2 = MatrixMultiply(m_pred_input,W_1)   ;
   
   
   CMatrixDouble pred_a_2 = MatrixSigmoid(pred_z_2);
   
   CMatrixDouble pred_z_3 = MatrixMultiply(m_a_2,W_2);
   
   CMatrixDouble pred_yHat = MatrixSigmoid(pred_z_3);
   
   
   return pred_yHat;


}

     
CMatrixDouble NeuralNetwork::Forward_Prop(CMatrixDouble &Input){




   m_input = Input;
   
   m_z_2 =  MatrixMultiply(m_input,W_1);          
    
   m_a_2 = MatrixSigmoid(m_z_2);
   
   m_z_3 = MatrixMultiply(m_a_2,W_2);
   
   CMatrixDouble yHat = MatrixSigmoid(m_z_3);
   
   
   
   return yHat;



}



void NeuralNetwork::Train(CMatrixDouble &Input,CMatrixDouble &correct_Val){

      bool Train_condition = true;
      y_cor = correct_Val;
      int iterations = 0 ;
      
      m_yHat= Forward_Prop(Input);
      
      
      ComputeDerivatives(Input,y_cor);
      
    
      
      CMatrixDouble mt_1(W_1.SizeRow(),W_1.SizeCol());
      mt_1=mt_1.Fill(0);
      
      
      CMatrixDouble mt_2(W_2.SizeRow(),W_2.SizeCol());
      mt_2=mt_2.Fill(0);
    
      double J = 0;
      while( Train_condition && iterations <m_maxiters){
   
    
            m_yHat= Forward_Prop(Input);
            ComputeDerivatives(Input,y_cor);
            
            
            J = Cost(Input,y_cor);
            
            
            
            if( J <m_alpha){
             Train_condition = false;
            }
        
       
         
       
        double beta_1 = m_beta_1;  
        double beta_2 = m_beta_2;
        
        CMatrixDouble beta_1_M_dJdW1 =dJdW1; 
        beta_1_M_dJdW1=beta_1_M_dJdW1.Fill(beta_1);
        CMatrixDouble beta_1_M_dJdW2 =dJdW2; 
        beta_1_M_dJdW2=beta_1_M_dJdW2.Fill(beta_1);
        CMatrixDouble ones_1_M_dJdW1 =dJdW1; 
        ones_1_M_dJdW1=ones_1_M_dJdW1.Fill(1);
        CMatrixDouble ones_1_M_dJdW2 =dJdW2; 
        ones_1_M_dJdW2=ones_1_M_dJdW2.Fill(1);
         
        mt_1 = beta_1_M_dJdW1*mt_1 +(ones_1_M_dJdW1-beta_1_M_dJdW1)*(dJdW1); 
        mt_2 = beta_1_M_dJdW2*mt_2 +(ones_1_M_dJdW2-beta_1_M_dJdW2)*(dJdW2);
        
        
        CMatrixDouble beta_2_mt_1 =mt_1; 
        beta_2_mt_1=beta_2_mt_1.Fill(beta_2);
        CMatrixDouble beta_2_mt_2 =mt_2; 
        beta_2_mt_2=beta_2_mt_2.Fill(beta_2);
        
        CMatrixDouble m_LearningRate_mt_1 = mt_1; 
        m_LearningRate_mt_1 = m_LearningRate_mt_1.Fill(m_LearningRate);
        CMatrixDouble m_LearningRate_mt_2 = mt_2; 
        m_LearningRate_mt_2 = m_LearningRate_mt_2.Fill(m_LearningRate);
        
        
        W_1 = W_1 - m_LearningRate_mt_1*( beta_2_mt_1*mt_1); 
        W_2 = W_2 - m_LearningRate_mt_2*( beta_2_mt_2*mt_2);
       
          
        iterations++;
                
   }
   
            
   if( m_verbose == true){  
   Print(iterations,"<<<< iterations");
   Print(J,"<<<< cost_value");
   }
   


}
     


double NeuralNetwork::Cost(CMatrixDouble &Input , CMatrixDouble &y_){

      CMatrixDouble X = Input;   
      CMatrixDouble Y = y_;
      m_yHat = Forward_Prop(X);
      
      CMatrixDouble temp = (Y -m_yHat);
      temp = temp*temp;  /// temp^2
      //Print(temp.SizeCol(),"===",temp.SizeRow());
      double J = .5*(temp.Sum()/(temp.SizeCol()*temp.SizeRow()) ); // 
      return J; 
}       





void NeuralNetwork::MatrixRandom(CMatrixDouble &m){

   

   for( int r=0 ; r<m.SizeRow(); r++){
   
      for( int c=0 ; c< m.SizeCol(); c++){
      
         m[r].Set(c, NormalDistSample(0,1));
      
      }
   
   
   
   }
   

}



double NeuralNetwork::Sigmoid(double x){


   return 1/(1+MathExp(-x));

}


double NeuralNetwork::Sigmoid_Prime(double x){


   return MathExp(-x)/(pow(1+MathExp(-x),2));

}





CMatrixDouble NeuralNetwork::MatrixSigmoid(CMatrixDouble &m){
   
   CMatrixDouble m_2(m.SizeRow(),m.SizeCol());; 
   for( int r=0 ; r<m.SizeRow(); r++){
   
      for( int c=0 ; c< m.SizeCol(); c++){
      
         m_2[r].Set(c, Sigmoid(m[r][c]));
      
      }
   
   
   
   }
   
   return m_2;

}



CMatrixDouble NeuralNetwork::MatrixSigmoidPrime(CMatrixDouble &m){
   
   CMatrixDouble m_2(m.SizeRow(),m.SizeCol());; 
   for( int r=0 ; r<m.SizeRow(); r++){
   
      for( int c=0 ; c< m.SizeCol(); c++){
      
         m_2[r].Set(c, Sigmoid_Prime(m[r][c]));
      
      }
   
   
   
   }
   
   return m_2;

}