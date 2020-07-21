package svm;

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import javax.imageio.ImageIO;

//import java.io.FileWriter;
//import java.io.BufferedWriter;

public class ReadData {
	
	public static final String TRAIN_IMAGES_FILE = "D:/mnist/train-images.idx3-ubyte";
    public static final String TRAIN_LABELS_FILE = "D:/mnist/train-labels.idx1-ubyte";
    public static final String TEST_IMAGES_FILE = "D:/mnist/t10k-images.idx3-ubyte";
    public static final String TEST_LABELS_FILE = "D:/mnist/t10k-labels.idx1-ubyte";

    /**
     * change bytes into a hex string.
     *
     * @param bytes bytes
     * @return the returned hex string
     */
    public static String bytesToHex(byte[] bytes) {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < bytes.length; i++) {
            String hex = Integer.toHexString(bytes[i] & 0xFF);
            if (hex.length() < 2) {
                sb.append(0);
            }
            sb.append(hex);
        }
        return sb.toString();
    }

    /**
     * get images of 'train' or 'test'
     *
     * @param fileName the file of 'train' or 'test' about image
     * @return one row show a `picture`
     */
    public static double[][] getImages(int num,String fileName) {
        double[][] x = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000803".equals(bytesToHex(bytes))) {                        // 读取魔数
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);           // 读取样本总数
                bin.read(bytes, 0, 4);
                int xPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每行所含像素点数
                bin.read(bytes, 0, 4);
                int yPixel = Integer.parseInt(bytesToHex(bytes), 16);           // 读取每列所含像素点数
                int n=num<number?num:number;
                x = new double[n][xPixel * yPixel];
                for (int i = 0; i < n; i++) {
                    double[] element = new double[xPixel * yPixel];
                    for (int j = 0; j < xPixel * yPixel; j++) {
                        //element[j] = bin.read();                                // 逐一读取像素值
                        // normalization
                        element[j] = bin.read() / 255.0;
                    }
                    x[i] = element;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return x;
    }

    /**
     * get labels of `train` or `test`
     *
     * @param fileName the file of 'train' or 'test' about label
     * @return
     */
    public static int[] getLabels(int num,String fileName) {
        int[] y = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000801".equals(bytesToHex(bytes))) {
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);
                int n=num<number?num:number;
                y = new int[n];
                for (int i = 0; i < n; i++) {
                    y[i] = bin.read();
                    if(y[i]==1) y[i]=1;
                    else y[i]=-1;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return y;
    }
    
    /**
     * draw a gray picture and the image format is JPEG.
     *
     * @param pixelValues pixelValues and ordered by column.
     * @param width       width
     * @param high        high
     * @param fileName    image saved file.
     */
    public static void drawGrayPicture(int[] pixelValues, int width, int high, String fileName) {
        BufferedImage bufferedImage = new BufferedImage(width, high, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < high; j++) {
                int pixel = 255 - pixelValues[i * high + j];
                int value = pixel + (pixel << 8) + (pixel << 16);   // r = g = b 时，正好为灰度
                bufferedImage.setRGB(j, i, value);
            }
        }
        try{
        	ImageIO.write(bufferedImage, "JPEG", new File(fileName));
        }
        catch(IOException e) {
        	System.out.println("IOException!");
        }
    }

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		SVM_SMO svm=new SVM_SMO();
		
		svm.train_sample_x=getImages(svm.num_train_sample,TRAIN_IMAGES_FILE);
		svm.train_sample_y=getLabels(svm.num_train_sample,TRAIN_LABELS_FILE);
		svm.test_sample_x=getImages(svm.num_test_sample,TEST_IMAGES_FILE);
		svm.test_sample_y=getLabels(svm.num_test_sample,TEST_LABELS_FILE);
		svm.initial();
		svm.SMOmain();
		double rate=svm.correct_rate();
		System.out.println("正确率"+rate);
	}

}
