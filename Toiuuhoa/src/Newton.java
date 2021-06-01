public class Newton {
    static Matrix2x2 mt = new Matrix2x2();

    // gradient fx
    public static double[][] gradient(double x1, double x2){
        double[][] rt = new double[2][1];
        rt[0][0] = -400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1);
        rt[1][0] = 200 * (x2 - x1 * x1);
        return rt;
    }

    // gradient 2 fx
    public static double[][] gradient2(double x1, double x2){
        double[][] rt = new double[2][2];
        rt[0][0] = 1200 * x1 * x1 - 400 * x2 + 2;
        rt[0][1] = rt[1][0] = -400 * x1;
        rt[1][1] = 200;
        return rt;
    }

    // Buoc newton
    public static double[][] stepNewton(double[] x) {
        // double[][] step = new double[2][1];
        double[][] ndgra2 = mt.nghichdao2x2(gradient2(x[0], x[1]));
        double[][] gra = gradient(x[0], x[1]);
        double[][] step = mt.nhan2matran(ndgra2, gra);
        for (int i = 0; i < step.length; i++){
            for (int j = 0; j < step[0].length; j++){
                step[i][j] *= -1;
            }
        }
        return step;
    }

    //Do giam Newton 
    public static double lamdamu2(double[] x ){
        // double rt = 0;
        double[][] ndgra2 = mt.nghichdao2x2(gradient2(x[0], x[1]));
        double[][] gra = gradient(x[0], x[1]);
        double[][] graT = mt.chuyenvi(gra);
        double[][] rt = mt.nhan2matran(mt.nhan2matran(graT, ndgra2), gra);
        return rt[0][0]; // chi co duy nhat rt[0][0]
    }

    //tinh ham Rosembrock 
    public static double fx(double x1, double x2) {
        return 100*Math.pow(x2-Math.pow(x1,2),2)+Math.pow(1-x1,2);
    }

    //Thuat toan Newton
    public static double[] newton(double alpha, double beta, double[] x, double t0){
        double[] rt = new double[2];
        int count = 0;
        double[][] step = stepNewton(x);
        double lamda = lamdamu2(x); //check
        while (count < 1000 || lamda > 2*Math.pow(10,-4)) {
            count++;
            step = stepNewton(x);
            while(fx(x[0] + t0 * step[0][0],x[1] + t0 * step[1][0]) > fx(x[0],x[1]) + t0 * alpha * lamda){
                t0 = beta * t0;
            }
            x[0] = x[0] + t0 * step[0][0];
            x[1] = x[1] + t0 * step[1][0];
            lamda = lamdamu2(x);
        }
        rt[0] = x[0];
        rt[1] = x[1];
        return rt;
    }

    public static void main(String[] args) {
        System.out.println("Bước ban đầu của thuật toán backtracking line search t0 = 1 ");
        double t0 = 1;
        System.out.println("Điểm xuất phát ban đầu x0 = (1.2, 1.2) và x0 = (-1.2,1)");
        double[] x01 = {1.2, 1.2};
        double[] x02 = {-1.2, 1};
        //dat alpha beta tuy y
        double alpha = 0.2;
        double beta = 0.5;

        //System.out.println(lamda(x01)); CHECK
        //System.out.println(stepNewton(x01));
        final long start1 = System.currentTimeMillis();
        double[] result1 = newton(alpha, beta, x01, t0);
        final long end1 = System.currentTimeMillis();
        System.out.println("run time = " + (end1 - start1) + " s");
        System.out.println("RESULT OF x01 = (" + result1[0] + ", " + result1[1] +")");

        System.out.println();

        final long start2 = System.currentTimeMillis();
        double[] result2 = newton(alpha, beta, x02, t0);
        final long end2 = System.currentTimeMillis();
        System.out.println("run time = " + (end2 - start2) + " s");
        System.out.println("RESULT OF x02 = (" + result2[0] + ", " + result2[1] +")");
    }
}
