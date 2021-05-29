public class GradientDescent {
    //ham tinh f(x0)
    public double fx(double[] xstart){
        return 100*Math.pow(xstart[1]-Math.pow(xstart[0],2),2)+Math.pow(1-xstart[0],2);
    }
    //ham tinh chuan euclid^2
    public double chuanEu(double[] x) {
        return Math.pow(x[0],2) + Math.pow(x[1],2);
    }
    // ham chay thuat toan Gradient Descent Backtracking line search
    public double[] backtrackinglinesearch(double alpha, double beta, double t0, double[] xstart) {
        double[] rt = new double[2];
        int count = 0;
        double[] gra = new double[2];
        double[] btGra = new double[2];
        double fxBtGra;
        double fx0;
        double gra2 = 99999;
        double btGra2;
        while (count < 1000 || Math.sqrt(gra2) > Math.pow(10,-4)){
            count++; //dem so buoc lap
            //update gradient f(x0)
            gra[0] = 400 * Math.pow(xstart[0],3) + 2*xstart[0] - 400 * xstart[0]*xstart[1] - 2;
            gra[1] = 200 * (xstart[1] - Math.pow(xstart[0], 2));
            //update bieu thuc x0 - t0 * gradient f(x0)
            btGra[0] = xstart[0] - t0 * gra[0]; 
            btGra[1] = xstart[1] - t0 * gra[1];
            //update gia tri cua ham Rosembrock voi x1 = btGra[0] va x2 = btGra[1]
            fxBtGra = fx(btGra);
            //update gia tri cua ham Rosembrock voi x1 = xstart[0] va x2 = xstart[1]
            fx0 = fx(xstart);
            //update chuan Euclid ^ 2
            gra2 = chuanEu(gra);
            //update bieu thuc f(x0) - alpha * t0 * gra2
            btGra2 = fx0 - alpha * t0 * gra2;
            
            while (btGra2 < fxBtGra) {
                t0 = beta * t0;

                btGra[0] = xstart[0] - t0 * gra[0]; 
                btGra[1] = xstart[1] - t0 * gra[1];

                fxBtGra = fx(btGra);

                btGra2 = fx0 - alpha * t0 * gra2;
            }
            xstart[0] = xstart[0] - t0 * gra[0];
            xstart[1] = xstart[1] - t0 * gra[1];
            count++;
        }
        rt[0] = xstart[0]; 
        rt[1] = xstart[1];
        return rt;
    }
    public static void main(String[] args) throws Exception {
        GradientDescent test = new GradientDescent();
        System.out.println("Bước ban đầu của thuật toán backtracking line search t0 = 1 ");
        double t0 = 1;
        System.out.println("Điểm xuất phát ban đầu x0 = (1.2, 1.2) và x0 = (-1.2,1)");
        double[] x01 = {1.2, 1.2};
        double[] x02 = {-1.2, 1};
        //dat alpha beta tuy y
        double alpha = 0.2;
        double beta = 0.5;
        
        final long start1 = System.currentTimeMillis();
        double[] result1 = test.backtrackinglinesearch(alpha, beta, t0, x01);
        final long end1 = System.currentTimeMillis();

        System.out.println("Running time = " + (end1 - start1) + " s");
        System.out.println("RESULT OF x01 = ("+ result1[0] + ", " + result1[1] +")");

        System.out.println();

        final long start2 = System.currentTimeMillis();
        double[] result2 = test.backtrackinglinesearch(alpha, beta, t0, x02);
        final long end2 = System.currentTimeMillis();
        
        System.out.println("Running time = " + (end2 - start2) + " s");
        System.out.println("RESULT OF x01 = ("+ result2[0] + ", " + result2[1] +")");
    }
}
