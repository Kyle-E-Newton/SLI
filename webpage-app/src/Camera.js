import React from 'react';
import Webcam from "react-webcam";
import axios from 'axios';

var url = "https://127.0.0.1:3000";
var val = 0;

export default class WebcamCapture extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            screenshot: null,
            tab: 0,
            start: false,
            intervalId: null
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
                val = makePostRequest(url + "/api/image", this.state.screenshot);
                displayValue(this.val);
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
            <button onClick={this.startOn}>Start</button>
            <button onClick={this.startOff}>Stop</button>
            {this.state.screenshot ? <img src ={this.state.screenshot} /> : null}

        </div>
        );
    }
}

async function makePostRequest(url, data) {
    const response = await axios.post(
        url,
        {data}
    );
    return response;
}

var displayValue = function(data) {
    //TODO: Show in text box
    console.log(data);
}

var onFailure = function() {
    console.error("Error in POST");
}
