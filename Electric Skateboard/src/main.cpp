/////////////////////////////////////////////////////////////////////////////////////////////////

// Including header files

#include "mbed.h"
#include "Servo.h"
#include "ultrasonic.h"
#include "ShiftBrite.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

// Set the pin connections

Serial pc(USBTX, USBRX);
Serial esp(p28, p27);       // Wi-Fi Module

Servo motor(p23);           // Motor

AnalogIn myDim(p17);        // Photo Cell
SPI spi1(p11, p12, p13);    // Spi 1 for Headlight
SPI spi2(p5, p6, p7);       // Spi 2 for Backlight
ShiftBrite myBrite1(p15,p16,spi1);  // Headlight
ShiftBrite myBrite2(p19,p20,spi2);  // Backlight


/////////////////////////////////////////////////////////////////////////////////////////////////

// Necessary variables for skateboard control besides wi-fi

bool ESC = false;       // ESC Initialization; need to be done only one time
bool night;             // True: Headlight On / False: Off
bool closeDist = false; // When object measured by sonar is close, becomes on
bool mode = true;       // Safety Mode - On: Auto-stop / Off: Keep going

/////////////////////////////////////////////////////////////////////////////////////////////////

// Necessary functions for skateboard control besides wi-fi


// Stop Function
void stop()
{
    myBrite2.Write(255,0,0);
    while (motor < 1.0) 
    {
        if (motor > 0.95)
            motor = 1.0;
        else
        {
            motor = motor + 0.01; // Brake
            wait(0.05);
            pc.printf("brake");
        } 
    }
    myBrite2.Write(0,0,0);
}


// Ultrasonic Function
void dist(int distance)
{
    //put code here to execute when the distance has changed
    pc.printf("Distance %d mm\r\n", distance);
    if(distance < 400) {
        closeDist=true;
    }
}


// Define ultrasonice pin since distance function is define!
ultrasonic sonic(p22, p18, .1, 1, &dist);


/////////////////////////////////////////////////////////////////////////////////////////////////

// Variables for Wi-Fi
char ssid[32] = "Verizon-SM-G900V-EC64";     // enter WiFi router ssid inside the quotes
char pwd [32] = "gsba591!"; // enter WiFi router password inside the quotes

char Vcc[10];
char Temp[10];

// things for sending/receiving ata over serial
volatile int tx_in=0;
volatile int tx_out=0;
volatile int rx_in=0;
volatile int rx_out=0;
const int buffer_size = 4095;
char tx_buffer[buffer_size+1];
char rx_buffer[buffer_size+1];
void Tx_interrupt();
void Rx_interrupt();
void send_line();
void read_line();

int count;
int DataRX;
int update;
char cmdbuff[1024];
char replybuff[4096];
char webdata[4096]; // This may need to be bigger depending on WEB browser used
char webbuff[4096];     // Currently using 1986 characters, Increase this if more web page data added
char timebuf[30];
void SendCMD(),getreply(),ReadWebData(),startserver();
void gettime(),setRTC();
char rx_line[1024];
int port        =80;  // set server port
int SERVtimeout =5;    // set server timeout in seconds in case link breaks.
struct tm t;
// manual set RTC values
int minute      =00;    // 0-59
int hour        =12;    // 2-23
int dayofmonth  =26;    // 1-31
int month       =8;     // 1-12
int year        =15;    // last 2 digits


/////////////////////////////////////////////////////////////////////////////////////////////////

// Function for Wi-Fi
void WiFi()
{
        if(DataRX==1) {
            ReadWebData();
            esp.attach(&Rx_interrupt, Serial::RxIrq);
        }
        if(update==1) // update time, hit count, and analog levels in the HUZZAH chip
        {
            // get new values
            gettime();
            count++;
            // send new values
            sprintf(cmdbuff, "count,time,analog1,analog2=%d,\"%s\",\"%s\",\"%s\"\r\n",count,timebuf,Temp,Vcc);
            SendCMD();
            getreply();
            update=0;   
        }
}


/////////////////////////////////////////////////////////////////////////////////////////////////

// Main Function
int main()
{
    // Wi-Fi Initialization
    pc.baud(9600);
    esp.baud(9600);
    esp.attach(&Rx_interrupt, Serial::RxIrq);
    esp.attach(&Tx_interrupt, Serial::TxIrq);
    if (time(NULL) < 1420070400) {
        setRTC();
    }
    startserver();
    DataRX=0;
    count=0;
    
    
    // Auto turn on/off for the headlight
    if (myDim*3.3 > 2.0)
    {
            night = false;    
            myBrite1.Write(0,0,0);  
    }
    else
    {
            myBrite1.Write(255,255,255);    
    }
    
    
    // Turn-off the backlight
    myBrite2.Write(0,0,0);
    
    
    // Motor ESC Initialization    
    if(ESC)
    {
            // Initialization Step for ESC; need to run only once
            motor = 1.0; //send longest PWM
            wait(3);
            motor = 0.5; //send longest PWM (negative scale)
            wait(3);
            motor = 0.0; //send shortest PWM
            wait(3);
    }
    else
    {
            // Send a signal to ESE to enable the process
            motor = 1;
            wait(3);    
    }
    
    
    // Start using sonar sensor    
    sonic.startUpdates();
    
    
    // Main control loop
    while(1) 
    {
        // Run Wi-Fi
        WiFi();
        
        // Check the distance
        sonic.checkDistance();
        
        // If object is near, run this code
        if(closeDist) 
        {      
            if(mode)
            {
                // Auto-stop
                pc.printf("\n Thread released \n ");
                stop();
                closeDist=false;
            }
            else
            {
                // Ignore
                closeDist=false;
            }
        } 
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// Sub-Functions for Wi-Fi


// Reads and processes GET and POST web data
void ReadWebData()
{
    wait_ms(200);
    esp.attach(NULL,Serial::RxIrq);
    DataRX=0;
    memset(webdata, '\0', sizeof(webdata));
    strcpy(webdata, rx_buffer);
    memset(rx_buffer, '\0', sizeof(rx_buffer));
    rx_in = 0;
    rx_out = 0;
    // check web data for form information
    if( strstr(webdata, "check=led1v") != NULL ) {
        if(motor > 0.0)
            motor = motor - 0.05;
        pc.printf("\nSpeed Up \n\r", motor);
    }
    if( strstr(webdata, "check=led2v") != NULL ) {
        if(motor < 1.0)
        {
            myBrite2.Write(255,0,0); 
            motor = motor + 0.05;
            myBrite2.Write(0,0,0); 
        }
        pc.printf("\nSpeed Down \n\r", motor);
    }
    if( strstr(webdata, "check=led3v") != NULL ) {
        mode=!mode;
        pc.printf("\Safety Mode Changed = %d \n\r", mode);
    }
    if( strstr(webdata, "check=led4v") != NULL ) {
        night=!night;
        if(night)
            myBrite1.Write(255,255,255); 
        else
            myBrite1.Write(0,0,0);
        pc.printf("\Night Mode = %d \n\r", night);
    }
    if( strstr(webdata, "check=led5v") != NULL ) {
        stop();
        pc.printf("\Speed Value = %d \n\r", motor);
    }
    if( strstr(webdata, "POST") != NULL ) { // set update flag if POST request
        update=1;
    }
    if( strstr(webdata, "GET") != NULL && strstr(webdata, "favicon") == NULL ) { // set update flag for GET request but do not want to update for favicon requests
        update=1;
    }
}
// Starts webserver
void startserver()
{
    gettime();
    pc.printf("++++++++++ Resetting ESP ++++++++++\r\n");
    strcpy(cmdbuff,"node.restart()\r\n");
    SendCMD();
    wait(2);
    getreply();
    
    pc.printf("\n++++++++++ Starting Server ++++++++++\r\n> ");

    // initial values
    sprintf(cmdbuff, "count,time,analog1,analog2=0,\"%s\",\"%s\",\"%s\"\r\n",timebuf,Temp,Vcc);
    SendCMD();
    getreply();
    wait(0.5);

    //create server
    sprintf(cmdbuff, "srv=net.createServer(net.TCP,%d)\r\n",SERVtimeout);
    SendCMD();
    getreply();
    wait(0.5);
    strcpy(cmdbuff,"srv:listen(80,function(conn)\r\n");
    SendCMD();
    getreply();
    wait(0.3);
        strcpy(cmdbuff,"conn:on(\"receive\",function(conn,payload) \r\n");
        SendCMD();
        getreply();
        wait(0.3);
        
        //print data to mbed
        strcpy(cmdbuff,"print(payload)\r\n");
        SendCMD();
        getreply();
        wait(0.2);
       
        //web page data
        strcpy(cmdbuff,"conn:send('<!DOCTYPE html><html><body><h1>Skateboard Wi-Fi Controller</h1>')\r\n");
        SendCMD();
        getreply();
        wait(0.4);
        strcpy(cmdbuff,"conn:send('<form method=\"POST\"')\r\n");
        SendCMD();
        getreply();
        wait(0.3);
        strcpy(cmdbuff, "conn:send('<p><input type=\"checkbox\" name=\"check\" value=\"led1v\"> Speed Up')\r\n");
        SendCMD();
        getreply();
        wait(0.3);
        strcpy(cmdbuff, "conn:send('<p><input type=\"checkbox\" name=\"check\" value=\"led2v\"> Speed Down')\r\n");
        SendCMD();
        getreply();
        wait(0.3);
        strcpy(cmdbuff, "conn:send('<p><input type=\"checkbox\" name=\"check\" value=\"led5v\"> Stop')\r\n");
        SendCMD();
        getreply();
        wait(0.3);
        strcpy(cmdbuff, "conn:send('<p><input type=\"checkbox\" name=\"check\" value=\"led3v\"> Safety Mode')\r\n");
        SendCMD();
        getreply();
        wait(0.3);
        strcpy(cmdbuff, "conn:send('<p><input type=\"checkbox\" name=\"check\" value=\"led4v\"> Night Mode')\r\n");
        SendCMD();
        getreply();
        wait(0.3);
        strcpy(cmdbuff,"conn:send('<p><input type=\"submit\" value=\"send-refresh\"></form>')\r\n");
        SendCMD();
        getreply();
        wait(0.3);
        strcpy(cmdbuff, "conn:send('<p><h2>How to use:</h2><ul><li>Select a checkbox to flip on/off</li><li>Click Send-Refresh to send data and refresh values</li></ul></body></html>')\r\n");
        SendCMD();
        getreply();
        wait(0.5); 
        // end web page data
        strcpy(cmdbuff, "conn:on(\"sent\",function(conn) conn:close() end)\r\n"); // close current connection
        SendCMD();
        getreply();
        wait(0.3);
        strcpy(cmdbuff, "end)\r\n");
        SendCMD();
        getreply();
        wait(0.2);
    strcpy(cmdbuff, "end)\r\n");
    SendCMD();
    getreply();
    wait(0.2);

    strcpy(cmdbuff, "tmr.alarm(0, 1000, 1, function()\r\n");
    SendCMD();
    getreply();
    wait(0.2);
    strcpy(cmdbuff, "if wifi.sta.getip() == nil then\r\n");
    SendCMD();
    getreply();
    wait(0.2);
    strcpy(cmdbuff, "print(\"Connecting to AP...\\n\")\r\n");
    SendCMD();
    getreply();
    wait(0.2);
    strcpy(cmdbuff, "else\r\n");
    SendCMD();
    getreply();
    wait(0.2);
    strcpy(cmdbuff, "ip, nm, gw=wifi.sta.getip()\r\n");
    SendCMD();
    getreply();
    wait(0.2);
    strcpy(cmdbuff,"print(\"IP Address: \",ip)\r\n");
    SendCMD();
    getreply();
    wait(0.2);
    strcpy(cmdbuff,"tmr.stop(0)\r\n");
    SendCMD();
    getreply();
    wait(0.2);
    strcpy(cmdbuff,"end\r\n");
    SendCMD();
    getreply();
    wait(0.2);
    strcpy(cmdbuff,"end)\r\n");
    SendCMD();
    getreply();
    wait(0.2);
    
    pc.printf("\n\n++++++++++ Ready ++++++++++\r\n\n");
}


// ESP Command data send
void SendCMD()
{
    int i;
    char temp_char;
    bool empty;
    i = 0;
// Start Critical Section - don't interrupt while changing global buffer variables
    NVIC_DisableIRQ(UART1_IRQn);
    empty = (tx_in == tx_out);
    while ((i==0) || (cmdbuff[i-1] != '\n')) {
// Wait if buffer full
        if (((tx_in + 1) % buffer_size) == tx_out) {
// End Critical Section - need to let interrupt routine empty buffer by sending
            NVIC_EnableIRQ(UART1_IRQn);
            while (((tx_in + 1) % buffer_size) == tx_out) {
            }
// Start Critical Section - don't interrupt while changing global buffer variables
            NVIC_DisableIRQ(UART1_IRQn);
        }
        tx_buffer[tx_in] = cmdbuff[i];
        i++;
        tx_in = (tx_in + 1) % buffer_size;
    }
    if (esp.writeable() && (empty)) {
        temp_char = tx_buffer[tx_out];
        tx_out = (tx_out + 1) % buffer_size;
// Send first character to start tx interrupts, if stopped
        esp.putc(temp_char);
    }
// End Critical Section
    NVIC_EnableIRQ(UART1_IRQn);
    return;
}

// Get Command and ESP status replies
void getreply()
{
    read_line();
    sscanf(rx_line,replybuff);
}
 
// Read a line from the large rx buffer from rx interrupt routine
void read_line() {
    int i;
    i = 0;
// Start Critical Section - don't interrupt while changing global buffer variables
    NVIC_DisableIRQ(UART1_IRQn);
// Loop reading rx buffer characters until end of line character
    while ((i==0) || (rx_line[i-1] != '\r')) {
// Wait if buffer empty
        if (rx_in == rx_out) {
// End Critical Section - need to allow rx interrupt to get new characters for buffer
            NVIC_EnableIRQ(UART1_IRQn);
            while (rx_in == rx_out) {
            }
// Start Critical Section - don't interrupt while changing global buffer variables
            NVIC_DisableIRQ(UART1_IRQn);
        }
        rx_line[i] = rx_buffer[rx_out];
        i++;
        rx_out = (rx_out + 1) % buffer_size;
    }
// End Critical Section
    NVIC_EnableIRQ(UART1_IRQn);
    rx_line[i-1] = 0;
    return;
}
 
 
// Interupt Routine to read in data from serial port
void Rx_interrupt() {
    DataRX=1;
    //led3=1;
// Loop just in case more than one character is in UART's receive FIFO buffer
// Stop if buffer full
    while ((esp.readable()) && (((rx_in + 1) % buffer_size) != rx_out)) {
        rx_buffer[rx_in] = esp.getc();
// Uncomment to Echo to USB serial to watch data flow
        pc.putc(rx_buffer[rx_in]);
        rx_in = (rx_in + 1) % buffer_size;
    }
    //led3=0;
    return;
}
 
 
// Interupt Routine to write out data to serial port
void Tx_interrupt() {
    //led2=1;
// Loop to fill more than one character in UART's transmit FIFO buffer
// Stop if buffer empty
    while ((esp.writeable()) && (tx_in != tx_out)) {
        esp.putc(tx_buffer[tx_out]);
        tx_out = (tx_out + 1) % buffer_size;
    }
    //led2=0;
    return;
}

void gettime()
{
    time_t seconds = time(NULL);
    strftime(timebuf,50,"%H:%M:%S %a %d %b %y", localtime(&seconds));
}

void setRTC()
{
    t.tm_sec = (0);             // 0-59
    t.tm_min = (minute);        // 0-59
    t.tm_hour = (hour);         // 0-23
    t.tm_mday = (dayofmonth);   // 1-31
    t.tm_mon = (month-1);       // 0-11  "0" = Jan, -1 added for Mbed RCT clock format
    t.tm_year = ((year)+100);   // year since 1900,  current DCF year + 100 + 1900 = correct year
    set_time(mktime(&t));       // set RTC clock
}

/////////////////////////////////////////////////////////////////////////////////////////////////