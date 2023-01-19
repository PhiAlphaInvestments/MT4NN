
#property strict 
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
// #define MacrosHello   "Hello, world!"
// #define MacrosYear    2010
//+------------------------------------------------------------------+
//| DLL imports                                                      |
//+------------------------------------------------------------------+
// #import "user32.dll"
//   int      SendMessageA(int hWnd,int Msg,int wParam,int lParam);
// #import "my_expert.dll"
//   int      ExpertRecalculate(int wParam,int lParam);
// #import
//+------------------------------------------------------------------+
//| EX5 imports                                                      |
//+------------------------------------------------------------------+
// #import "stdlib.ex5"
//   string ErrorDescription(int error_code);
// #import
//+------------------------------------------------------------------

//inspired by,
//https://github.com/stephencwelch/Neural-Networks-Demystified




class CRowDouble;

int ArrayResizeAL(CRowDouble &arr[],const int size)
  {
//---
   return(ArrayResize(arr,size));
  }
int ArrayResizeAL(double &arr[],const int size)
  {
//---
   return(ArrayResize(arr,size));
  }


class CRowDouble
  {
private:
   double            m_array[];

public:
                     CRowDouble(void);
                    ~CRowDouble(void);
   //--- methods
   int               Size(void) const;
   void              Resize(const int n);
   void              Set(const int i,const double d);
   //--- overloading
   double            operator[](const int i) const;
   void              operator=(const double &array[]);
   void              operator=(const CRowDouble &r);
  };
//+------------------------------------------------------------------+
//| Constructor without parameters                                   |
//+------------------------------------------------------------------+
CRowDouble::CRowDouble(void)
  {

  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CRowDouble::~CRowDouble(void)
  {

  }
//+------------------------------------------------------------------+
//| Row size                                                         |
//+------------------------------------------------------------------+
int CRowDouble::Size(void) const
  {
   return(ArraySize(m_array));
  }
//+------------------------------------------------------------------+
//| Resize                                                           |
//+------------------------------------------------------------------+
void CRowDouble::Resize(const int n)
  {
   ArrayResizeAL(m_array,n);
  }
//+------------------------------------------------------------------+
//| Set value                                                        |
//+------------------------------------------------------------------+
void CRowDouble::Set(const int i,const double d)
  {
   m_array[i]=d;
  }
//+------------------------------------------------------------------+
//| Indexing operator                                                |
//+------------------------------------------------------------------+
double CRowDouble::operator[](const int i) const
  {
   return(m_array[i]);
  }
//+------------------------------------------------------------------+
//| Overloading (=)                                                  |
//+------------------------------------------------------------------+
void CRowDouble::operator=(const double &array[])
  {
   int size=ArraySize(array);
//--- check
   if(size==0)
      return;
//--- filling array
   ArrayResizeAL(m_array,size);
   for(int i=0;i<size;i++)
      m_array[i]=array[i];
  }
//+------------------------------------------------------------------+
//| Overloading (=)                                                  |
//+------------------------------------------------------------------+
void CRowDouble::operator=(const CRowDouble &r)
  {
   int size=r.Size();
//--- check
   if(size==0)
      return;
//--- filling array
   ArrayResizeAL(m_array,size);
   for(int i=0;i<size;i++)
      m_array[i]=r[i];
  }
//+------------------------------------------------------------------+
//| Matrix (double)                                                  |
//+------------------------------------------------------------------+
class CMatrixDouble
  {
private:
   //--- array
   CRowDouble        m_rows[];
   int               Col;
   int               Row;
   CRowDouble        temp[];
public:
   //--- constructors, destructor
                     CMatrixDouble(void);
                     CMatrixDouble(const int rows);
                     CMatrixDouble(const int rows,const int cols);
                     CMatrixDouble(const CMatrixDouble &m); 
                    ~CMatrixDouble(void);
   //--- methods
   int               SizeRow(void) const;
   int               SizeCol(void) const;
   void              Resize(const int n,const int m);
   CMatrixDouble     Transpose(void);
   void              PrintMatrix(void);
   //--- overloading
   CRowDouble       *operator[](const int i) const;
   void              operator=(const CMatrixDouble &m);
   CMatrixDouble             operator+(const CMatrixDouble &m);
   CMatrixDouble              operator*(const CMatrixDouble &m);
   CMatrixDouble              operator-(const CMatrixDouble &m);
  };
//+------------------------------------------------------------------+
//| Constructor without parameters                                   |
//+------------------------------------------------------------------+
CMatrixDouble::CMatrixDouble(void)
  {
  Row = 0;
  Col = 0;

  }
//+------------------------------------------------------------------+
//| Constructor with one parameter                                   |
//+------------------------------------------------------------------+
CMatrixDouble::CMatrixDouble(const int rows)
  {
   ArrayResizeAL(m_rows,rows);
   Row = rows;
   Col = 0; 
  }
//+------------------------------------------------------------------+
//| Constructor with two parameters                                  |
//+------------------------------------------------------------------+
CMatrixDouble::CMatrixDouble(const int rows,const int cols)
  {
  Row = rows;
  Col = cols; 
   ArrayResizeAL(m_rows,rows);
   for(int i=0;i<rows;i++)
      m_rows[i].Resize(cols);
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CMatrixDouble::~CMatrixDouble(void)
  {

  }
//+------------------------------------------------------------------+
//| Get size                                                         |
//+------------------------------------------------------------------+
int CMatrixDouble::SizeRow(void) const
  {
   return(Row);
  }
  
  
int CMatrixDouble::SizeCol(void) const
  {
   return(Col);
  }  
//+------------------------------------------------------------------+
//| Resize                                                           |
//+------------------------------------------------------------------+
void CMatrixDouble::Resize(const int n,const int m)
  {
//--- check
   if(n<0 || m<0)
      return;
//--- change sizes
   ArrayResizeAL(m_rows,n);
   for(int i=0;i<n;i++)
      m_rows[i].Resize(m);
   Row = n;
   Col = m;
  }
//+------------------------------------------------------------------+
//| Indexing operator                                                |
//+------------------------------------------------------------------+
CRowDouble *CMatrixDouble::operator[](const int i) const
  {
   return(GetPointer(m_rows[i]));
  }
//+------------------------------------------------------------------+
//| Overloading (=)                                                  |
//+------------------------------------------------------------------+
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
//--- check
   if(r==0)
      return;
   int c=m[0].Size();
//--- check
   if(c==0)
      return;
//--- change size
   ArrayResizeAL(m_rows,r);
   for(int i=0;i<r;i++)
      m_rows[i].Resize(c);
//--- copy
   for(int i=0;i<r;i++)
      m_rows[i]=m[i];
  }

void CMatrixDouble::CMatrixDouble(const CMatrixDouble &m){
   Row = m.SizeRow();
   Col = m.SizeCol();

   int r=m.SizeRow();
//--- check
   if(r==0)
      return;
   int c=m[0].Size();
//--- check
   if(c==0)
      return;
//--- change size
   ArrayResizeAL(m_rows,r);
   for(int i=0;i<r;i++)
      m_rows[i].Resize(c);
//--- copy
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



 
class NN_volatility{

   protected:
      
      int m_deep;
      int m_depth;
      string m_Symbol;
      double close[];
      CMatrixDouble m_input;
      CMatrixDouble m_pred_input;
      ENUM_TIMEFRAMES m_TF;
      CMatrixDouble m_z_2;
      CMatrixDouble m_a_2;
      CMatrixDouble m_z_3;
      CMatrixDouble m_yHat;
      double y_cor;
      double m_alpha;
            
   
      public:
   
   CMatrixDouble W_1;
   CMatrixDouble W_2;
   CMatrixDouble W_1_LSTM;
   NN_volatility(string Symbol_, ENUM_TIMEFRAMES TimeFrame, int History_Depth, int Number_of_Neurons, double alpha);
   double Sigmoid(double x);
   double Sigmoid_Prime(double x);
   int    Sgn( double Value);
   void   MatrixRandom(CMatrixDouble& m);
   CMatrixDouble   MatrixSigmoid(CMatrixDouble& m);
   CMatrixDouble   MatrixSigmoidPrime(CMatrixDouble& m);
   CMatrixDouble   Forward_Prop();
   CMatrixDouble   MatrixMultiply( CMatrixDouble& m1, CMatrixDouble& m2); 
   double   Cost();
   void     UpdateValues(int shift);
   void     Train(int shift);
   double   Prediction();
   void     Tester();
};


CMatrixDouble NN_volatility::MatrixMultiply( CMatrixDouble& m1, CMatrixDouble& m2){

    
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
    
    Print( "Failed Dot Product");
    m1.PrintMatrix();
    m2.PrintMatrix();
    }  
    /* 
    
  
      */
        
      
         
   
      return A;







}


NN_volatility::NN_volatility(string Symbol_,ENUM_TIMEFRAMES TimeFrame,int History_Depth,int Number_of_Neurons,double alpha){




   m_Symbol = Symbol_;
   m_depth = History_Depth;
   m_deep  = Number_of_Neurons;
   m_TF    = TimeFrame;
   m_alpha = alpha;
   
   
   CMatrixDouble random_LSTM(1,m_deep);
   CMatrixDouble random_W1(m_depth,m_deep);
   CMatrixDouble random_W2(m_deep,1);
   
   
   
   MatrixRandom(random_W1);
   MatrixRandom(random_W2);
   MatrixRandom(random_LSTM);
   
   W_1_LSTM.Resize(1,m_deep);
   W_1.Resize(m_depth,m_deep);
   W_2.Resize(m_deep,1);
  
   
   W_1 = random_W1;
   W_2 = random_W2;
   W_1_LSTM = random_LSTM;
   
   
   
   
   ArrayResize(close,m_depth+5,0);
   
   m_yHat.Resize(1,1);
   m_yHat[0].Set(0,3.141);
   y_cor = -1;
   

}


void NN_volatility::UpdateValues(int shift){

   for( int i =0 ; i< m_depth+5;i++){
   
      close[i] = iClose(m_Symbol,m_TF,i+shift);//(Sgn(iClose(m_Symbol,m_TF,i+shift)-iClose(m_Symbol,m_TF,i+shift+1)) +1)/2;
   }
   
   m_input.Resize(1,m_depth);
   
   for( int i=0 ; i< m_depth; i++){
   
       m_input[0].Set(i,close[i+2] - close[i+1+2]);  // m_input[0][i-1] = close[i] - close[i+1];
   }
   
   
   
   m_pred_input.Resize(1,m_depth);
   
   for( int i=0 ; i< m_depth; i++){
   
      m_pred_input[0].Set(i, close[i+1+0]-close[i+1+1]);
   }
   
   y_cor = (Sgn(close[0]-close[1])+1)/2;
}


double NN_volatility::Prediction(void){

   CMatrixDouble pred_z_2 =  MatrixMultiply(m_pred_input,W_1)+ W_1_LSTM;// + W_1_LSTM;// m_pred_input.MatMul(W_1)+ W_1_LSTM;
   
   
   CMatrixDouble pred_a_2 = MatrixSigmoid(pred_z_2);
   
   CMatrixDouble pred_z_3 =  MatrixMultiply(pred_a_2, W_2);
   
   CMatrixDouble pred_yHat = MatrixSigmoid(pred_z_3);

   return pred_yHat[0][0];

}

CMatrixDouble NN_volatility::Forward_Prop(void){

   

   m_z_2 =  MatrixMultiply(m_input,W_1)+ W_1_LSTM; //+ W_1_LSTM;
   
   
  
   m_a_2 = MatrixSigmoid(m_z_2);
   
   m_z_3 = MatrixMultiply(m_a_2,W_2); 
      
   m_yHat = MatrixSigmoid(m_z_3);
   
   
   return m_yHat;
   
   


}
void NN_volatility::Train(int shift){

   bool Train_condition = true;
   UpdateValues(shift);
   while(Train_condition){
   
      m_yHat = Forward_Prop();
      double J = Cost();
      
      if (J < m_alpha){ 
         Train_condition = false;
      
      }
   
      CMatrixDouble X_m_matrix(1,1); 
      X_m_matrix[0].Set(0,-1*(y_cor-m_yHat[0][0]));
      
      CMatrixDouble cost = X_m_matrix;
      
      CMatrixDouble z_3_prime = MatrixSigmoidPrime(m_z_3);
      
      CMatrixDouble delta3 =  MatrixMultiply(cost,z_3_prime);//cost.MatMul(z_3_prime);
      
      
      
      CMatrixDouble dJdW2  =  MatrixMultiply(m_a_2.Transpose(),delta3); ///m_a_2.Transpose().MatMul(delta3);
      
      
      CMatrixDouble z_2_prime = MatrixSigmoidPrime(m_z_2);
      
      
      
      
      CMatrixDouble delta2 =  MatrixMultiply(delta3,W_2.Transpose())*z_2_prime; //delta3.MatMul(W_2.Transpose())*z_2_prime;
      
      
      
      CMatrixDouble dJdW1 =  MatrixMultiply(m_input.Transpose(),delta2);//m_input.Transpose().MatMul(delta2);
      
      W_1= W_1 - dJdW1;
      W_2= W_2 - dJdW2;
      
   
   
   
   
   }

   W_1_LSTM = MatrixMultiply(m_input,W_1);//.Transpose();//m_input.MatMul(W_1);
   W_1_LSTM = W_1_LSTM +W_1_LSTM;

}



double NN_volatility::Cost(void){


   double J = .5*pow( y_cor - m_yHat[0][0],2);

   return J;
}



void NN_volatility::Tester(){
   
   //Print(m_yHat[0][0]);
   Print(NormalDistSample(0,1));

}


void NN_volatility::MatrixRandom(CMatrixDouble &m){

   

   for( int r=0 ; r<m.SizeRow(); r++){
   
      for( int c=0 ; c< m.SizeCol(); c++){
      
         m[r].Set(c, NormalDistSample(0,1));
      
      }
   
   
   
   }
   

}



double NN_volatility::Sigmoid(double x){


   return 1/(1+MathExp(-x));

}


double NN_volatility::Sigmoid_Prime(double x){


   return MathExp(-x)/(pow(1+MathExp(-x),2));

}


int NN_volatility::Sgn(double Value){
   
     int RES;
     
     if ( Value>0){
         
        RES = 1;    
     
     }
      else{
      
         RES=-1;
      
      }

   return RES;
}



CMatrixDouble NN_volatility::MatrixSigmoid(CMatrixDouble &m){
   
   CMatrixDouble m_2(m.SizeRow(),m.SizeCol());; 
   for( int r=0 ; r<m.SizeRow(); r++){
   
      for( int c=0 ; c< m.SizeCol(); c++){
      
         m_2[r].Set(c, Sigmoid(m[r][c]));
      
      }
   
   
   
   }
   
   return m_2;

}



CMatrixDouble NN_volatility::MatrixSigmoidPrime(CMatrixDouble &m){
   
   CMatrixDouble m_2(m.SizeRow(),m.SizeCol());; 
   for( int r=0 ; r<m.SizeRow(); r++){
   
      for( int c=0 ; c< m.SizeCol(); c++){
      
         m_2[r].Set(c, Sigmoid_Prime(m[r][c]));
      
      }
   
   
   
   }
   
   return m_2;

}
