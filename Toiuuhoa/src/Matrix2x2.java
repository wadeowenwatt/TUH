public class Matrix2x2 {
    double[][] mt2x2 = new  double[2][2];
    
    public Matrix2x2() {

    }

    public Matrix2x2(double[][] mt2x2) {
        this.mt2x2 = mt2x2;
    }

    // ham tinh det(A) 
    public double det(double[][] mt) {
        return mt[0][0] * mt[1][1] - mt[0][1] * mt[1][0];
    }
    /* tim A^-1 (ma tran nghich dao) (2x2) 
                                  1     | d      -b |                
    A = | a   b |   => A^-1 = --------- |           |
        | c   d |              ad - bc  | -c      a |

    */
    public double[][] nghichdao2x2(double[][] mt){
        if (det(mt) == 0) return null;
        double t = 1 / (mt[0][0]*mt[1][1]-mt[0][1]*mt[1][0]);
    
        double[][] mtnd = new double[2][2];
        mtnd[0][0] = mt[1][1] * t; 
        mtnd[0][1] = -1 * mt[0][1] * t;
        mtnd[1][0] = -1 * mt[1][0] * t;
        mtnd[1][1] = mt[0][0] * t;
        return mtnd;
    }
    // tim ma tran chuyen vi
    public double[][] chuyenvi(double[][] a) {
        double[][] cv = new double[a[0].length][a.length];
        for (int i = 0; i < cv.length; i++){
            for (int j = 0; j < cv[0].length; j++){
                cv[i][j] = a[j][i];
            }
        }
        return cv;
    }
    //nhan 2 ma tran
    public double[][] nhan2matran(double[][] a, double[][] b) {
        double[][] c = new double[a.length][b[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                for (int k = 0; k < a[0].length; k++) { 
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        } 
        return c;
    }

}
