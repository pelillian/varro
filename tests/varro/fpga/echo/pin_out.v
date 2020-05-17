module sinx (input [11:0] digital, output [5:0] analog, input clk); 

    assign analog[5:0] = digital[5:0];    

//    initial begin
//        while(1) begin
//        end
//            sum = 0;  
//            x = digital; 
//            for (n = 0; n < 5; n = n + 1) begin
//                sum = sum + (((-1) ^ n) * (x^(2*n + 1)) / factorial(2 * n + 1));
//            end
//            analog = $realtobits(sum);  
//    end
    
endmodule
