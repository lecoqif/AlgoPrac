import java.util.Stack;

public class MyQueue {
    Stack<Integer> st1;
    Stack<Integer> st2;
    public MyQueue() {
        st1 = new Stack<>();
        st2 = new Stack<>();
    }

    /** Push element x to the back of queue. */
    public void push(int x) {
        st1.push(x);
    }

    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
        if(st2.isEmpty()){
            while(!st1.isEmpty()){
                int x = st1.pop();
                st2.push(x);
            }
        }
        return st2.pop();

    }

    /** Get the front element. */
    public int peek() {
        if(st2.isEmpty()){
            while(!st1.isEmpty()){
                int x = st1.pop();
                st2.push(x);
            }
        }
        return st2.peek();
    }

    /** Returns whether the queue is empty. */
    public boolean empty() {
        return st1.isEmpty() && st2.isEmpty();
    }
}
