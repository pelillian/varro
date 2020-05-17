module sinx (input [11:0] digital, output [5:0] analog, input clk); 

    integer x = 64'd0;  
    reg [2:0] n = 3'b0; 

    reg state = 1'b0; 
    
    always@ (posedge clk) begin
        analog[5:0] = digital[5:0];
    end

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
