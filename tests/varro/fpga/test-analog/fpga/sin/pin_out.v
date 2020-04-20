module sinx (input [11:0] digital, output [5:0] analog); 

    function automatic [64:0] factorial; 
        input [64:0] number;
        begin
            if (number < 2) 
                factorial = 1; 
            else
                factorial = number * factorial(number - 1);  
        end
    endfunction

    reg [64:0] sum = 64'b1;  
    integer x = 64'd1;  
    integer n = 64'd1; 

    always @(posedge clk) begin

    end        
//    initial begin
//        forever begin
//            sum = 0;  
//            x = digital; 
//            for (n = 0; n < 5; n = n + 1) begin
//                sum = sum + (((-1) ^ n) * (x^(2*n + 1)) / factorial(2 * n + 1));
//            end
//            analog = $realtobits(sum);  
//        end
//    end
    
endmodule
