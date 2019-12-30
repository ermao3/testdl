package demo;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by DongBin on 2019/2/15.
 */
public class CreateLine {

    public static void main(String[] args) throws Exception {


        createLine(false);

    }

    public static List<Double> createLine(boolean needError) {
        //System.out.println("11111111111111111111111111111");

        boolean isError = false;
        int errorNum  = 10;
        List<Double> list   = new ArrayList<Double>();
        double start = Math.random()*100.00;
        list.add(start);
        int direction = 1;
        for(int i=0;i<500;i++){
            double change = ((Math.random()*5.00)/errorNum)*direction;
            start = start+change;
            //System.out.println(start);

            list.add(start);
            int x =   (int)(1+Math.random()*(20-1+1));
            if(x==10){
                direction=-direction;
            }
            if(needError){
                if(!isError){
                    int k =   (int)(1+Math.random()*(150-1+1));
                    if(k==10){
                        isError=true;

                    }
                }else{
                    int k =   (int)(1+Math.random()*(40-1+1));
                    if(k==10){
                        isError=false;

                    }
                }
            }


            if(isError){
                errorNum = 1;
            }else {
                errorNum = 10;
            }
        }
        //System.out.println("2222222222222222222222222222222222222");
        return list;
    }
}
