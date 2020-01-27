import java.util.ArrayDeque;
import java.util.LinkedList;

public class MovingAverage {
    LinkedList<Integer> arr;
    int size;
    int total;
    int elements;

    public MovingAverage(int size) {
        this.size = size;
        total = 0;
        elements = 0;
        arr = new LinkedList<>();
    }

    public double next(int val) {
        elements++;
        if(elements > size){
            int x = arr.removeFirst();
            total -= x;
            elements--;
        }
        total += val;
        arr.add(val);
        return (double) total / elements;
    }
}
