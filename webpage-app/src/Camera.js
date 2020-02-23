import React, { useCallback } from 'react';
import Webcam from "react-webcam";

var url = "http://127.0.0.1:5000";
var val = 0;

export default class WebcamCapture extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            screenshot: null,
            tab: 0,
            start: false,
            intervalId: null,
            textMessage: ""
        };
    }
    
    setRef = webcam => {
        this.webcam = webcam;
    };
      
    capture = () => {
        const screenshot = this.webcam.getScreenshot();
        this.setState({ screenshot })
    };

    timer = () => {
        if(this.start){
            this.intervalId = setInterval(() => {
                this.capture();
                val = makePostRequest(url + "/api/image", this.state.screenshot, this.onSuccess);
                //console.log(val);
                //displayValue(this.val);
            }, 2000);
        }else{
            clearInterval(this.intervalId);
        }
    }

    startOn = () => {
        this.start = true;
        this.timer();
    }

    startOff = () => {
        this.start = false;
        this.timer();
    }

     onSuccess = (data) => {
        var char = data[11];
        console.log(data);
        console.log(char);
        this.setState({
            textMessage: this.state.textMessage + char
        });
    }
      
    render() {
    const videoConstraints = {
        width: 1280,
        height: 720,
        facingMode: "user"
        };
      
    return (
        <div>
            <Webcam id="cam"
                audio={false}
                ref={this.setRef}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                />
            <button  id = "startbutton" onClick={this.startOn}>Start</button>
            <button id = "stopbutton"onClick={this.startOff}>Stop</button>
            {this.state.screenshot ? <img id = "snap" src ={this.state.screenshot} /> : null}
            <textarea id = "resultarea" multiline={true} value={this.state.textMessage} style={{ fontSize: 32 }}></textarea>
        </div>
        );
    }
}

 function makePostRequest(url, data, success) {
    var xhr = new XMLHttpRequest();

    xhr.onreadystatechange = function() {
        if(xhr.readyState == 4) {
            success(xhr.response);
        }
    }
    xhr.open('POST', url, false);
    xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8')
    xhr.send(JSON.stringify({"data": data}))
}

var displayValue = function(data) {
    //TODO: Show in text box
    console.log(data);
}
/*
var onSuccess = function(data) {
    var char = data["letter"];
    console.log(data);
    this.setState({
        textMessage: this.state.textMessage + char
    });
}
*/
var onFailure = function() {
    console.error("Error in POST");
}
